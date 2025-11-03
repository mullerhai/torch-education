package torch.edu

import torch.*
import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn as nn
import scala.collection.mutable.ListBuffer
import scala.math

// ATKT模型实现
class ATKT[ParamType <: FloatNN: Default](
    joint: Boolean,
    mask_future: Boolean,
    length: Int,
    num_skills: Int,
    skill_dim: Int,
    answer_dim: Int,
    hidden_dim: Int,
    attention_dim: Int = 80,
    epsilon: Float = 10.0f,
    beta: Float = 0.2f,
    dropout: Double = 0.2
) extends TensorModule[ParamType] with HasParams[ParamType] {
  // 模型参数
  val skill_dim_val = skill_dim
  val answer_dim_val = answer_dim
  val hidden_dim_val = hidden_dim
  val num_skills_val = num_skills
  val epsilon_val = epsilon
  val beta_val = beta
  
  // LSTM层
  val rnn = torch.nn.LSTM[ParamType](skill_dim + answer_dim, hidden_dim, batch_first = true)
  
  // Dropout层
  val dropout_layer = torch.nn.Dropout[ParamType](dropout)
  
  // 全连接层
  val fc = torch.nn.Linear[ParamType](hidden_dim * 2, num_skills)
  
  // Sigmoid激活函数
  val sig = torch.nn.Sigmoid[ParamType]()
  
  // 嵌入层
  val skill_emb = torch.nn.Embedding[ParamType](num_skills + 1, skill_dim)
  val answer_emb = torch.nn.Embedding[ParamType](2 + 1, answer_dim)
  
  // 初始化嵌入层最后一个权重为0
  skill_emb.weight.slice(0, num_skills).fill_(0.0f)
  answer_emb.weight.slice(0, 2).fill_(0.0f)
  
  // 注意力机制相关层
  val mlp = torch.nn.Linear[ParamType](hidden_dim, attention_dim)
  val similarity = torch.nn.Linear[ParamType](attention_dim, 1, bias = false)
  
  // 损失函数
  val loss_fn = nn.BCELoss()
  
  // 获取所有参数
  override def params: Seq[Tensor[ParamType]] = {
    Seq(rnn, dropout_layer, fc, skill_emb, answer_emb, mlp, similarity)
  }.flatMap(_.params)
  
  // apply方法调用forward
  override def apply(t: Tensor[ParamType]*): Tensor[ParamType] = {
    // 简化处理，假设输入是feed_dict
    val feed_dict = Map[String, Tensor[ParamType]](
      "skills" -> t(0),
      "responses" -> t(1)
    )
    forward(feed_dict)"pred".asInstanceOf[Tensor[ParamType]]
  }
  
  // 注意力模块
  def attention_module(lstm_output: Tensor[ParamType]): Tensor[ParamType] = {
    // 计算注意力权重
    val att_w = mlp(lstm_output)
    val tanh_att_w = torch.tanh(att_w)
    val similarity_att_w = similarity(tanh_att_w)
    
    // 创建上三角掩码
    val device = lstm_output.device
    val seq_len = lstm_output.shape(1)
    val attn_mask = ut_mask(seq_len, device)
    
    // 扩展注意力权重并应用掩码
    val expanded_att_w = similarity_att_w.transpose(1, 2).expand(lstm_output.shape(0), seq_len, seq_len).clone()
    val masked_att_w = expanded_att_w.masked_fill(attn_mask, Float.NegativeInfinity)
    
    // 计算注意力分布
    val alphas = F.softmax(masked_att_w, dim = -1)
    
    // 应用注意力
    val attn_output = torch.bmm(alphas, lstm_output)
    
    // 计算累积注意力输出
    val attn_output_cum = torch.cumsum(attn_output, dim = 1)
    val attn_output_cum_1 = attn_output_cum - attn_output
    
    // 拼接最终输出
    val final_output = torch.cat(Seq(attn_output_cum_1, lstm_output), dim = 2)
    
    final_output
  }
  
  // 前向传播
  def forward(feed_dict: Map[String, Tensor[ParamType]], perturbation: Option[Tensor[ParamType]] = None): Map[String, Any] = {
    val c = feed_dict("skills")
    val r = feed_dict("responses")
    
    // 处理掩码
    val masked_r = r * (r > (-1.0f)).long()
    
    // 获取技能和答案序列
    val skill = c.slice(1, 0, -length)
    val answer = masked_r.slice(1, 0, -length)
    
    // 嵌入层
    val skill_embedding = skill_emb(skill.long())
    val answer_embedding = answer_emb(answer.long())
    
    // 拼接嵌入
    val skill_answer = torch.cat(Seq(skill_embedding, answer_embedding), dim = 2)
    val answer_skill = torch.cat(Seq(answer_embedding, skill_embedding), dim = 2)
    
    // 根据答案选择不同的嵌入组合
    val expanded_answer = answer.unsqueeze(2).expand(skill_answer.shape)
    val skill_answer_embedding = torch.where(
      expanded_answer === 1.0f,
      skill_answer,
      answer_skill
    )
    
    // 保存原始特征用于对抗训练
    val skill_answer_embedding1 = skill_answer_embedding.clone()
    
    // 应用扰动（如果有）
    val input_with_perturbation = perturbation match {
      case Some(pert) => skill_answer_embedding + pert
      case None => skill_answer_embedding
    }
    
    // LSTM前向传播
    val (out, _) = rnn(input_with_perturbation)
    
    // 应用注意力机制
    val attn_out = attention_module(out)
    
    // 全连接层和dropout
    val output = fc(dropout_layer(attn_out))
    
    // 处理joint模式
    val res = if (joint) {
      val seq_len = output.shape(1)
      val mid = seq_len / 2
      // 扩展中间部分的输出
      val expanded_output = output.slice(1, mid, mid+1).expand(-1, seq_len - mid, -1)
      val modified_output = output.slice(1, 0, mid).cat(expanded_output, dim = 1)
      sig(modified_output)
    } else {
      sig(output)
    }
    
    // 计算预测结果
    val (preds, true_val) = if (joint) {
      val one_hot_c = torch.nn.functional.one_hot(c.slice(1, length).long(), num_skills)
      val joint_preds = (res * one_hot_c).sum(-1)
      val seq_len = res.shape(1)
      val mid = seq_len / 2
      val rshft = r.slice(1, length)
      (joint_preds.slice(1, mid), rshft.slice(1, mid))
    } else {
      val one_hot_c = torch.nn.functional.one_hot(c.slice(1, length).long(), num_skills)
      val normal_preds = (res * one_hot_c).sum(-1)
      (normal_preds, r.slice(1, length))
    }
    
    // 处理mask_future模式
    val (final_preds, final_true) = if (mask_future) {
      (preds.slice(1, -length), r.slice(1, -length))
    } else {
      (preds, true_val)
    }
    
    // 返回结果
    Map(
      "pred" -> final_preds,
      "true" -> final_true,
      "features" -> skill_answer_embedding1
    )
  }
  
  // 损失计算，包含对抗训练部分
  def loss(feed_dict: Map[String, Tensor[ParamType]], out_dict: Map[String, Any]): (Tensor[ParamType], Int, Double) = {
    val pred = out_dict("pred").asInstanceOf[Tensor[ParamType]].flatten()
    val true_val = out_dict("true").asInstanceOf[Tensor[ParamType]].flatten()
    val features = out_dict("features").asInstanceOf[Tensor[ParamType]]
    
    // 计算掩码
    val mask = true_val > (-1.0f)
    
    // 计算原始损失
    val loss_val = loss_fn(pred.masked_select(mask), true_val.masked_select(mask))
    
    // 计算对抗扰动
    val features_grad = torch.autograd.grad(Seq(loss_val), Seq(features), retain_graph = true)
    val p_adv = _l2_normalize_adv(features_grad.head.data) * epsilon_val
    
    // 应用对抗扰动并计算对抗损失
    val new_out_dict = forward(feed_dict, Some(p_adv))
    val pred_res = new_out_dict("pred").asInstanceOf[Tensor[ParamType]].flatten()
    val adv_loss = loss_fn(pred_res.masked_select(mask), true_val.masked_select(mask))
    
    // 组合损失
    val combined_loss = loss_val + (beta_val * adv_loss)
    
    (combined_loss, pred.masked_select(mask).numel(), true_val.masked_select(mask).sum().double()())
  }
  
  // 上三角掩码函数
  def ut_mask(seq_len: Int, device: torch.Device): Tensor[Boolean] = {
    torch.triu(torch.ones(seq_len, seq_len), diagonal = 1).to(device)
  }
  
  // L2归一化函数
  def _l2_normalize_adv(d: Tensor[ParamType]): Tensor[ParamType] = {
    val d_np = d.CPU().toArray().asInstanceOf[Array[Float]]
    val shape = d.shape
    val reshaped_d = d_np.grouped(shape(1) * shape(2)).toArray()
    
    val normalized_d = reshaped_d.map { batch =>
      val norm = math.sqrt(batch.map(x => x * x).sum()) + 1e-16f
      batch.map(x => x / norm)
    }
    
    Tensor[ParamType](normalized_d.flatten(), shape)
  }
  
  // 切换训练模式
  def withTrainingMode(training: Boolean): ATKT[ParamType] = {
    this.train(training)
    this
  }
}

// 伴生对象提供工厂方法
object ATKT {
  def apply[ParamType <: FloatNN: Default](
      joint: Boolean,
      mask_future: Boolean,
      length: Int,
      num_skills: Int,
      skill_dim: Int,
      answer_dim: Int,
      hidden_dim: Int,
      attention_dim: Int = 80,
      epsilon: Float = 10.0f,
      beta: Float = 0.2f,
      dropout: Double = 0.2
  ): ATKT[ParamType] = {
    new ATKT[ParamType](joint, mask_future, length, num_skills, skill_dim, answer_dim, hidden_dim, attention_dim, epsilon, beta, dropout)
  }
}
