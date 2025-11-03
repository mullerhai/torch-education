package torch.edu

import torch.*
import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}

import scala.collection.mutable.ListBuffer
import scala.math

// CL4KTTransformerLayer的推断实现
class CL4KTTransformerLayer[ParamType <: FloatNN: Default](
    d_model: Int,
    d_feature: Int,
    d_ff: Int,
    n_heads: Int,
    dropout: Double,
    kq_same: Boolean
) extends TensorModule[ParamType] with HasParams[ParamType] {
  // 实现multi-head attention
  private val W_Q = torch.nn.Linear[ParamType](d_model, n_heads * d_feature)
  private val W_K = if (kq_same) W_Q else torch.nn.Linear[ParamType](d_model, n_heads * d_feature)
  private val W_V = torch.nn.Linear[ParamType](d_model, n_heads * d_feature)
  private val W_O = torch.nn.Linear[ParamType](n_heads * d_feature, d_model)
  
  // 实现feed forward
  private val W_1 = torch.nn.Linear[ParamType](d_model, d_ff)
  private val W_2 = torch.nn.Linear[ParamType](d_ff, d_model)
  
  private val layer_norm1 = torch.nn.LayerNorm[ParamType](d_model)
  private val layer_norm2 = torch.nn.LayerNorm[ParamType](d_model)
  
  private val drop = torch.nn.Dropout[ParamType](dropout)
  private val drop2 = torch.nn.Dropout[ParamType](dropout)
  
  // 获取所有参数
  override def params: Seq[Tensor[ParamType]] = 
    Seq(W_Q, W_K, W_V, W_O, W_1, W_2, layer_norm1, layer_norm2).flatMap(_.params)
  
  // apply方法调用forward
  override def apply(t: Tensor[ParamType]*): Tensor[ParamType] = {
    // 根据Python代码中的调用方式，这里假设参数顺序为mask, query, key, values, apply_pos
    val mask = t(0)
    val query = t(1)
    val key = t(2)
    val values = t(3)
    val apply_pos = t(4).bools()
    
    forward(mask, query, key, values, apply_pos)._1
  }
  
  def forward(
      mask: Tensor[ParamType],
      query: Tensor[ParamType],
      key: Tensor[ParamType],
      values: Tensor[ParamType],
      apply_pos: Boolean
  ): (Tensor[ParamType], Tensor[ParamType]) = {
    val batch_size = query.shape(0)
    val seq_length = query.shape(1)
    
    // multi-head attention
    var q = W_Q(query).view(batch_size, seq_length, n_heads, d_feature).transpose(1, 2)
    var k = W_K(key).view(batch_size, seq_length, n_heads, d_feature).transpose(1, 2)
    var v = W_V(values).view(batch_size, seq_length, n_heads, d_feature).transpose(1, 2)
    
    // 计算注意力分数
    var scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(d_feature.double())
    
    // 应用mask
    if (mask.shape(0) > 0) {
      val expanded_mask = mask.unsqueeze(1).unsqueeze(2)
      scores = scores.masked_fill(expanded_mask === 0, Float.NegativeInfinity)
    }
    
    // 计算注意力权重
    val attn = F.softmax(scores, dim = -1)
    val attn_dropped = drop(attn)
    
    // 应用注意力
    var context = torch.matmul(attn_dropped, v)
    
    // 重塑
    context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
    
    // 应用输出变换
    context = W_O(context)
    context = drop2(context)
    
    // 残差连接和层归一化
    context = layer_norm1(query + context)
    
    // feed forward
    var output = W_2(F.gelu(W_1(context)))
    output = drop(output)
    output = layer_norm2(context + output)
    
    (output, attn)
  }
}

// Similarity类实现
class Similarity[ParamType <: FloatNN: Default](temp: Double) extends TensorModule[ParamType] with HasParams[ParamType] {
  private val cos = torch.nn.CosineSimilarity[ParamType](dim = -1)
  private val tempValue = temp
  
  override def params: Seq[Tensor[ParamType]] = Seq.empty
  
  override def apply(t: Tensor[ParamType]*): Tensor[ParamType] = {
    val x = t(0)
    val y = t(1)
    forward(x, y)
  }
  
  def forward(x: Tensor[ParamType], y: Tensor[ParamType]): Tensor[ParamType] = {
    cos(x, y) / tempValue
  }
}

// CL4KT主模型
class CL4KT[ParamType <: FloatNN: Default](
    joint: Boolean,
    mask_response: Boolean,
    pred_last: Boolean,
    mask_future: Boolean,
    length: Int,
    trans: Boolean,
    num_skills: Int,
    num_questions: Int,
    seq_len: Int,
    args: Map[String, Any]
) extends TensorModule[ParamType] with HasParams[ParamType] {
  // 初始化参数
  val hidden_size: Int = args("hidden_size").asInstanceOf[Int]
  val num_blocks: Int = args("num_blocks").asInstanceOf[Int]
  val num_attn_heads: Int = args("num_attn_heads").asInstanceOf[Int]
  val kq_same: Boolean = args("kq_same").asInstanceOf[Boolean]
  val final_fc_dim: Int = args("final_fc_dim").asInstanceOf[Int]
  val d_ff: Int = args("d_ff").asInstanceOf[Int]
  val l2: Double = args("l2").asInstanceOf[Double]
  val dropout: Double = args("dropout").asInstanceOf[Double]
  val reg_cl: Double = args("reg_cl").asInstanceOf[Double]
  val negative_prob: Double = args("negative_prob").asInstanceOf[Double]
  val hard_negative_weight: Double = args("hard_negative_weight").asInstanceOf[Double]
  
  // 嵌入层
  val question_embed = torch.nn.Embedding[ParamType](num_skills + 2, hidden_size, padding_idx = 0)
  val interaction_embed = torch.nn.Embedding[ParamType](2 * (num_skills + 2), hidden_size, padding_idx = 0)
  val sim = new Similarity[ParamType](args("temp").asInstanceOf[Double])
  
  // Transformer层
  val question_encoder = ListBuffer[CL4KTTransformerLayer[ParamType]]()
  val interaction_encoder = ListBuffer[CL4KTTransformerLayer[ParamType]]()
  val knoweldge_retriever = ListBuffer[CL4KTTransformerLayer[ParamType]]()
  
  // 初始化编码器层
  for (_ <- 0 until num_blocks) {
    question_encoder += new CL4KTTransformerLayer[ParamType](
      d_model = hidden_size,
      d_feature = hidden_size / num_attn_heads,
      d_ff = d_ff,
      n_heads = num_attn_heads,
      dropout = dropout,
      kq_same = kq_same
    )
    
    interaction_encoder += new CL4KTTransformerLayer[ParamType](
      d_model = hidden_size,
      d_feature = hidden_size / num_attn_heads,
      d_ff = d_ff,
      n_heads = num_attn_heads,
      dropout = dropout,
      kq_same = kq_same
    )
    
    knoweldge_retriever += new CL4KTTransformerLayer[ParamType](
      d_model = hidden_size,
      d_feature = hidden_size / num_attn_heads,
      d_ff = d_ff,
      n_heads = num_attn_heads,
      dropout = dropout,
      kq_same = kq_same
    )
  }
  
  // 输出层
  val out = if (trans) {
    torch.nn.Sequential[ParamType](
      torch.nn.Linear[ParamType](2 * hidden_size, final_fc_dim),
      torch.nn.GELU[ParamType](),
      torch.nn.Dropout[ParamType](dropout),
      torch.nn.Linear[ParamType](final_fc_dim, final_fc_dim / 2),
      torch.nn.GELU[ParamType](),
      torch.nn.Dropout[ParamType](dropout),
      torch.nn.Linear[ParamType](final_fc_dim / 2, num_skills)
    )
  } else {
    torch.nn.Sequential[ParamType](
      torch.nn.Linear[ParamType](2 * hidden_size, final_fc_dim),
      torch.nn.GELU[ParamType](),
      torch.nn.Dropout[ParamType](dropout),
      torch.nn.Linear[ParamType](final_fc_dim, final_fc_dim / 2),
      torch.nn.GELU[ParamType](),
      torch.nn.Dropout[ParamType](dropout),
      torch.nn.Linear[ParamType](final_fc_dim / 2, 1)
    )
  }
  
  // 损失函数
  val cl_loss_fn = torch.nn.CrossEntropyLoss()
  val loss_fn = torch.nn.BCELoss()
  
  // 获取所有参数
  override def params: Seq[Tensor[ParamType]] = {
    Seq(question_embed, interaction_embed, out) ++
    question_encoder ++ interaction_encoder ++ knoweldge_retriever
  }.flatMap(_.params)
  
  // apply方法调用forward
  override def apply(t: Tensor[ParamType]*): Tensor[ParamType] = {
    // 这里简化处理，假设输入是batch数据
    val batch = Map[String, Tensor[ParamType]](
      "skills" -> t(0),
      "responses" -> t(1),
      "attention_mask" -> t(2)
    )
    forward(batch)"pred".asInstanceOf[Tensor[ParamType]]
  }
  
  // 获取交互嵌入
  def get_interaction_embed(skills: Tensor[ParamType], responses: Tensor[ParamType]): Tensor[ParamType] = {
    val masked_responses = responses * (responses > (-1.0f)).long()
    val interactions = skills + (num_skills * masked_responses)
    interaction_embed(interactions.long())
  }
  
  // 前向传播
  def forward(batch: Map[String, Any]): Map[String, Any] = {
    if (isTraining) {
      // 训练模式
      val q_i = batch("skills").asInstanceOf[Tuple3[Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]]]._1
      val q_j = batch("skills").asInstanceOf[Tuple3[Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]]]._2
      val q = batch("skills").asInstanceOf[Tuple3[Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]]]._3
      
      val r_i = batch("responses").asInstanceOf[Tuple4[Tensor[ParamType], Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]]]._1
      val r_j = batch("responses").asInstanceOf[Tuple4[Tensor[ParamType], Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]]]._2
      val r = batch("responses").asInstanceOf[Tuple4[Tensor[ParamType], Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]]]._3
      val neg_r = batch("responses").asInstanceOf[Tuple4[Tensor[ParamType], Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]]]._4
      
      val attention_mask_i = batch("attention_mask").asInstanceOf[Tuple3[Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]]]._1
      val attention_mask_j = batch("attention_mask").asInstanceOf[Tuple3[Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]]]._2
      val attention_mask = batch("attention_mask").asInstanceOf[Tuple3[Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]]]._3
      
      var r_input: Tensor[ParamType] = null
      var cshft: Tensor[ParamType] = null
      
      // 根据不同模式处理数据
      if (trans) {
        cshft = q.slice(1, length)
        q_i = q_i.slice(1, 0, -length)
        q_j = q_j.slice(1, 0, -length)
        q = q.slice(1, 0, -length)
        r_i = r_i.slice(1, 0, -length)
        r_j = r_j.slice(1, 0, -length)
        r_input = r.slice(1, 0, -length)
        neg_r = neg_r.slice(1, 0, -length)
        attention_mask_i = attention_mask_i.slice(1, 0, -length)
        attention_mask_j = attention_mask_j.slice(1, 0, -length)
        attention_mask = attention_mask.slice(1, 0, -length)
      } else if (mask_future) {
        attention_mask = attention_mask.masked_fill(
          torch.arange(attention_mask.shape(1)).unsqueeze(0).to(attention_mask.device) >= 
          (attention_mask.shape(1) - length),
          0.0f
        )
        q_i = q_i * attention_mask
        q_j = q_j * attention_mask
        q = q * attention_mask
        r_i = r_i * attention_mask
        r_j = r_j * attention_mask
        r_input = r * attention_mask
        neg_r = neg_r * attention_mask
        attention_mask_i = attention_mask_i * attention_mask
        attention_mask_j = attention_mask_j * attention_mask
      } else if (mask_response) {
        attention_mask = attention_mask.masked_fill(
          torch.arange(attention_mask.shape(1)).unsqueeze(0).to(attention_mask.device) >= 
          (attention_mask.shape(1) - length),
          0.0f
        )
        r_i = r_i * attention_mask
        r_j = r_j * attention_mask
        r_input = r * attention_mask
        neg_r = neg_r * attention_mask
        attention_mask_i = attention_mask_i * attention_mask
        attention_mask_j = attention_mask_j * attention_mask
      } else {
        r_input = r
      }
      
      // 计算嵌入
      val ques_i_embed = question_embed(q_i.long())
      val ques_j_embed = question_embed(q_j.long())
      val inter_i_embed = get_interaction_embed(q_i, r_i)
      val inter_j_embed = get_interaction_embed(q_j, r_j)
      
      var inter_k_embed: Tensor[ParamType] = null
      if (negative_prob > 0) {
        inter_k_embed = get_interaction_embed(q, neg_r)
      }
      
      // BERT双向注意力
      var ques_i_score = ques_i_embed
      var ques_j_score = ques_j_embed
      var inter_i_score = inter_i_embed
      var inter_j_score = inter_j_embed
      var inter_k_score: Tensor[ParamType] = null
      
      // 应用question encoder
      for (block <- question_encoder) {
        val (output_i, _) = block.forward(
          mask = torch.ones(1), // mask=2表示双向注意力
          query = ques_i_score,
          key = ques_i_score,
          values = ques_i_score,
          apply_pos = false
        )
        ques_i_score = output_i
        
        val (output_j, _) = block.forward(
          mask = torch.ones(1),
          query = ques_j_score,
          key = ques_j_score,
          values = ques_j_score,
          apply_pos = false
        )
        ques_j_score = output_j
      }
      
      // 应用interaction encoder
      for (block <- interaction_encoder) {
        val (output_i, _) = block.forward(
          mask = torch.ones(1),
          query = inter_i_score,
          key = inter_i_score,
          values = inter_i_score,
          apply_pos = false
        )
        inter_i_score = output_i
        
        val (output_j, _) = block.forward(
          mask = torch.ones(1),
          query = inter_j_score,
          key = inter_j_score,
          values = inter_j_score,
          apply_pos = false
        )
        inter_j_score = output_j
        
        if (negative_prob > 0) {
          val (output_k, _) = block.forward(
            mask = torch.ones(1),
            query = inter_k_embed,
            key = inter_k_embed,
            values = inter_k_embed,
            apply_pos = false
          )
          inter_k_score = output_k
        }
      }
      
      // 池化计算
      val pooled_ques_i_score = (ques_i_score * attention_mask_i.unsqueeze(-1)).sum(1) / 
        attention_mask_i.sum(-1).unsqueeze(-1)
      val pooled_ques_j_score = (ques_j_score * attention_mask_j.unsqueeze(-1)).sum(1) / 
        attention_mask_j.sum(-1).unsqueeze(-1)
      
      // 计算对比学习损失
      val ques_cos_sim = sim(pooled_ques_i_score.unsqueeze(1), pooled_ques_j_score.unsqueeze(0))
      val ques_labels = torch.arange(ques_cos_sim.shape(0)).to(ques_cos_sim.device)
      val question_cl_loss = cl_loss_fn(ques_cos_sim, ques_labels.long())
      
      val pooled_inter_i_score = (inter_i_score * attention_mask_i.unsqueeze(-1)).sum(1) / 
        attention_mask_i.sum(-1).unsqueeze(-1)
      val pooled_inter_j_score = (inter_j_score * attention_mask_j.unsqueeze(-1)).sum(1) / 
        attention_mask_j.sum(-1).unsqueeze(-1)
      
      var inter_cos_sim = sim(pooled_inter_i_score.unsqueeze(1), pooled_inter_j_score.unsqueeze(0))
      var pooled_inter_k_score: Tensor[ParamType] = null
      var neg_inter_cos_sim: Tensor[ParamType] = null
      
      if (negative_prob > 0) {
        pooled_inter_k_score = (inter_k_score * attention_mask.unsqueeze(-1)).sum(1) / 
          attention_mask.sum(-1).unsqueeze(-1)
        neg_inter_cos_sim = sim(pooled_inter_i_score.unsqueeze(1), pooled_inter_k_score.unsqueeze(0))
        inter_cos_sim = torch.cat(Seq(inter_cos_sim, neg_inter_cos_sim), dim = 1)
      }
      
      val inter_labels = torch.arange(inter_cos_sim.shape(0)).to(inter_cos_sim.device)
      
      if (negative_prob > 0) {
        // 添加硬负样本权重
        val weights = torch.zeros(inter_cos_sim.shape)
        for (i <- 0 until neg_inter_cos_sim.shape(1)) {
          weights.slice(0, i, i+1).slice(1, inter_cos_sim.shape(1) - neg_inter_cos_sim.shape(1) + i, inter_cos_sim.shape(1) - neg_inter_cos_sim.shape(1) + i + 1) = 
            hard_negative_weight
        }
        inter_cos_sim = inter_cos_sim + weights
      }
      
      val interaction_cl_loss = cl_loss_fn(inter_cos_sim, inter_labels.long())
      
      // 预测部分
      val q_embed = question_embed(q.long())
      val i_embed = get_interaction_embed(q, r_input)
      
      var x = q_embed
      var y = i_embed
      
      for (block <- question_encoder) {
        val (output, _) = block.forward(
          mask = torch.zeros(1), // mask=1表示单向注意力
          query = x,
          key = x,
          values = x,
          apply_pos = true
        )
        x = output
      }
      
      for (block <- interaction_encoder) {
        val (output, _) = block.forward(
          mask = torch.zeros(1),
          query = y,
          key = y,
          values = y,
          apply_pos = true
        )
        y = output
      }
      
      var attn: Tensor[ParamType] = null
      for (block <- knoweldge_retriever) {
        val (output, attention) = block.forward(
          mask = torch.zeros(1), // mask=0表示无mask
          query = x,
          key = x,
          values = y,
          apply_pos = true
        )
        x = output
        attn = attention
      }
      
      // 生成预测
      val retrieved_knowledge = torch.cat(Seq(x, q_embed), dim = -1)
      var output: Tensor[ParamType] = null
      var true_output: Tensor[ParamType] = null
      
      if (trans) {
        if (joint) {
          output = out(retrieved_knowledge)
          val seq_len = output.shape(1)
          val mid = seq_len / 2
          // 扩展中间部分的输出
          val expanded_output = output.slice(1, mid, mid+1).expand(-1, seq_len - mid, -1)
          output = output.slice(1, 0, mid).cat(expanded_output, dim = 1)
          output = torch.sigmoid(output)
          // 使用one-hot编码
          val one_hot_cshft = torch.nn.functional.one_hot(cshft.long(), num_skills)
          output = (output * one_hot_cshft).sum(-1)
          val rshft = r.slice(1, length)
          true_output = rshft.slice(1, mid)
          output = output.slice(1, mid)
        } else {
          output = torch.sigmoid(out(retrieved_knowledge))
          val one_hot_cshft = torch.nn.functional.one_hot(cshft.long(), num_skills)
          output = (output * one_hot_cshft).sum(-1)
          true_output = r.slice(1, length)
        }
      } else if (mask_future || pred_last || mask_response) {
        output = torch.sigmoid(out(retrieved_knowledge)).squeeze(-1)
        output = output.slice(1, -length)
        true_output = r.slice(1, -length)
      } else {
        output = torch.sigmoid(out(retrieved_knowledge)).squeeze(-1)
        output = output.slice(1, length)
        true_output = r.slice(1, length)
      }
      
      // 返回结果
      Map(
        "pred" -> output,
        "true" -> true_output,
        "cl_loss" -> (question_cl_loss + interaction_cl_loss),
        "attn" -> attn
      )
    } else {
      // 推理模式
      val q = batch("skills").asInstanceOf[Tensor[ParamType]]
      val r = batch("responses").asInstanceOf[Tensor[ParamType]]
      val attention_mask = batch("attention_mask").asInstanceOf[Tensor[ParamType]]
      
      var r_input: Tensor[ParamType] = null
      var cshft: Tensor[ParamType] = null
      
      // 根据不同模式处理数据
      if (trans) {
        cshft = q.slice(1, length)
        q = q.slice(1, 0, -length)
        r_input = r.slice(1, 0, -length)
        r_input = (r_input > (-1.0f)) * r_input
        attention_mask = attention_mask.slice(1, 0, -length)
      } else if (mask_future) {
        attention_mask = attention_mask.masked_fill(
          torch.arange(attention_mask.shape(1)).unsqueeze(0).to(attention_mask.device) >= 
          (attention_mask.shape(1) - length),
          0.0f
        )
        q = q * attention_mask
        r_input = r * attention_mask
      } else if (mask_response) {
        attention_mask = attention_mask.masked_fill(
          torch.arange(attention_mask.shape(1)).unsqueeze(0).to(attention_mask.device) >= 
          (attention_mask.shape(1) - length),
          0.0f
        )
        r_input = r * attention_mask
      } else {
        r_input = r
      }
      
      // 前向传播
      val q_embed = question_embed(q.long())
      val i_embed = get_interaction_embed(q, r_input)
      
      var x = q_embed
      var y = i_embed
      
      for (block <- question_encoder) {
        val (output, _) = block.forward(
          mask = torch.zeros(1),
          query = x,
          key = x,
          values = x,
          apply_pos = true
        )
        x = output
      }
      
      for (block <- interaction_encoder) {
        val (output, _) = block.forward(
          mask = torch.zeros(1),
          query = y,
          key = y,
          values = y,
          apply_pos = true
        )
        y = output
      }
      
      var attn: Tensor[ParamType] = null
      for (block <- knoweldge_retriever) {
        val (output, attention) = block.forward(
          mask = torch.zeros(1),
          query = x,
          key = x,
          values = y,
          apply_pos = true
        )
        x = output
        attn = attention
      }
      
      // 生成预测
      val retrieved_knowledge = torch.cat(Seq(x, q_embed), dim = -1)
      var output: Tensor[ParamType] = null
      var true_output: Tensor[ParamType] = null
      
      if (trans) {
        if (joint) {
          output = out(retrieved_knowledge)
          val seq_len = output.shape(1)
          val mid = seq_len / 2
          // 扩展中间部分的输出
          val expanded_output = output.slice(1, mid, mid+1).expand(-1, seq_len - mid, -1)
          output = output.slice(1, 0, mid).cat(expanded_output, dim = 1)
          output = torch.sigmoid(output)
          // 使用one-hot编码
          val one_hot_cshft = torch.nn.functional.one_hot(cshft.long(), num_skills)
          output = (output * one_hot_cshft).sum(-1)
          val rshft = r.slice(1, length)
          true_output = rshft.slice(1, mid)
          output = output.slice(1, mid)
        } else {
          output = torch.sigmoid(out(retrieved_knowledge))
          val one_hot_cshft = torch.nn.functional.one_hot(cshft.long(), num_skills)
          output = (output * one_hot_cshft).sum(-1)
          true_output = r.slice(1, length)
        }
      } else if (mask_future || pred_last || mask_response) {
        output = torch.sigmoid(out(retrieved_knowledge)).squeeze(-1)
        output = output.slice(1, -length)
        true_output = r.slice(1, -length)
      } else {
        output = torch.sigmoid(out(retrieved_knowledge)).squeeze(-1)
        output = output.slice(1, length)
        true_output = r.slice(1, length)
      }
      
      // 返回结果
      Map(
        "pred" -> output,
        "true" -> true_output,
        "attn" -> attn,
        "x" -> x
      )
    }
  }
  
  // 损失计算
  def loss(feed_dict: Map[String, Any], out_dict: Map[String, Any]): (Tensor[ParamType], Int, Double) = {
    val pred = out_dict("pred").asInstanceOf[Tensor[ParamType]].flatten()
    val true_val = out_dict("true").asInstanceOf[Tensor[ParamType]].flatten()
    val cl_loss = out_dict("cl_loss").asInstanceOf[Tensor[ParamType]].mean()
    
    val mask = true_val > (-1.0f)
    val loss_val = if (cl_loss.isNaN().any().toBoolean) {
      loss_fn(pred.masked_select(mask), true_val.masked_select(mask))
    } else {
      loss_fn(pred.masked_select(mask), true_val.masked_select(mask)) + (reg_cl * cl_loss)
    }
    
    (loss_val, pred.masked_select(mask).numel(), true_val.masked_select(mask).sum().double()())
  }
  
  // 工厂方法，方便创建实例
  def withTrainingMode(training: Boolean): CL4KT[ParamType] = {
    this.train(training)
    this
  }
}

// 伴生对象提供工厂方法
object CL4KT {
  def apply[ParamType <: FloatNN: Default](
      joint: Boolean,
      mask_response: Boolean,
      pred_last: Boolean,
      mask_future: Boolean,
      length: Int,
      trans: Boolean,
      num_skills: Int,
      num_questions: Int,
      seq_len: Int,
      args: Map[String, Any]
  ): CL4KT[ParamType] = {
    new CL4KT[ParamType](joint, mask_response, pred_last, mask_future, length, trans, num_skills, num_questions, seq_len, args)
  }
}
