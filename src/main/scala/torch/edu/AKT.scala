package torch.edu

import torch.*
import torch.nn.{functional as F, *}
import torch.nn.modules.{HasParams, TensorModule}

import scala.collection.mutable.ListBuffer
import scala.math
import scala.util.Try

// 对应Python中的Dim枚举类
enum Dim:
  case batch, seq, feature

// MultiHeadAttention类的实现
class MultiHeadAttention[ParamType <: FloatNN: Default](
    embedding_size: Int,
    d_feature: Int,
    n_heads: Int,
    dropout: Double,
    kq_same: Boolean,
    bias: Boolean = true
) extends TensorModule[ParamType] with HasParams[ParamType]:
  import torch.ops.nn.init.{constant_, xavierUniform_}

  val d_k: Int = d_feature
  val h: Int = n_heads
  val kq_same_flag: Boolean = kq_same

  val v_linear: Linear[ParamType] = nn.Linear(embedding_size, embedding_size, bias = bias)
  val k_linear: Linear[ParamType] = nn.Linear(embedding_size, embedding_size, bias = bias)
  val q_linear: Option[Linear[ParamType]] = if (!kq_same_flag) Some(Linear(embedding_size, embedding_size, bias = bias)) else None
  val dropout_layer: Dropout[ParamType] = Dropout(dropout)
  val proj_bias: Boolean = bias
  val out_proj: Linear[ParamType] = Linear(embedding_size, embedding_size, bias = bias)
  val gammas: Tensor[ParamType] = torch.zeros(n_heads, 1, 1).requiresGrad_(true)
  xavierUniform_(gammas)

  // 初始化参数
  _reset_parameters()

  private def _reset_parameters(): Unit = {
    xavierUniform_(k_linear.weight)
    xavierUniform_(v_linear.weight)
    if (!kq_same_flag) q_linear.foreach(l => xavierUniform_(l.weight))

    if (proj_bias) {
      constant_(k_linear.bias.get, 0.0)
      constant_(v_linear.bias.get, 0.0)
      if (!kq_same_flag) q_linear.foreach(l => constant_(l.bias.get, 0.0))
      constant_(out_proj.bias.get, 0.0)
    }
  }

  def forward(q: Tensor[ParamType], k: Tensor[ParamType], v: Tensor[ParamType], mask: Tensor[ParamType], zero_pad: Boolean, pdiff: Option[Tensor[ParamType]] = None): Tensor[ParamType] = {
    val bs = q.size(0)

    // 线性变换并分割为多头
    var k_proj = k_linear(k).view(bs, -1, h, d_k)
    var q_proj = if (!kq_same_flag) q_linear.get(q).view(bs, -1, h, d_k) else k_linear(q).view(bs, -1, h, d_k)
    var v_proj = v_linear(v).view(bs, -1, h, d_k)

    // 转置以获得维度 bs * h * sl * embedding_size
    k_proj = k_proj.transpose(1, 2)
    q_proj = q_proj.transpose(1, 2)
    v_proj = v_proj.transpose(1, 2)

    // 计算注意力
    val scores = attention(q_proj, k_proj, v_proj, d_k, mask, dropout_layer, zero_pad, Some(gammas), pdiff)

    // 连接多头并通过最终线性层
    val concat = scores.transpose(1, 2).contiguous().view(bs, -1, embedding_size)
    val output = out_proj(concat)

    output
  }

  // apply方法调用forward
  def apply(q: Tensor[ParamType], k: Tensor[ParamType], v: Tensor[ParamType], mask: Tensor[ParamType], zero_pad: Boolean, pdiff: Option[Tensor[ParamType]] = None): Tensor[ParamType] = {
    forward(q, k, v, mask, zero_pad, pdiff)
  }

  def pad_zero(scores: Tensor[ParamType], bs: Int, dim: Int, zero_pad: Boolean, device: Device): Tensor[ParamType] = {
    if (zero_pad) {
      val padZero = torch.zeros(bs, 1, dim).to(device)
      torch.cat(Seq(padZero, scores.slice(1, 0, -1)), dim = 1)
    } else {
      scores
    }
  }

// attention函数的实现
private def attention[ParamType <: FloatNN: Default](
    q: Tensor[ParamType],
    k: Tensor[ParamType],
    v: Tensor[ParamType],
    d_k: Int,
    mask: Tensor[ParamType],
    dropout: Dropout[ParamType],
    zero_pad: Boolean,
    gamma: Option[Tensor[ParamType]] = None,
    pdiff: Option[Tensor[ParamType]] = None
): Tensor[ParamType] = {
  // 计算注意力分数
  val scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k.double())
  val bs = scores.size(0)
  val head = scores.size(1)
  val seqlen = scores.size(2)
  val device = q.device

  // 创建位置索引
  val x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
  val x2 = x1.transpose(0, 1).contiguous()

  // 计算位置效应
  val distScores = withNoGrad {
    val scores_ = scores.masked_fill(mask == torch.zeros(1), -1e32f)
    val softmaxScores = F.softmax(scores_, dim = -1)
    val maskedScores = softmaxScores * mask
    val distcumScores = torch.cumsum(maskedScores, dim = -1)
    val disttotalScores = torch.sum(maskedScores, dim = -1, keepdim = true)
    val positionEffect = torch.abs(x1 - x2).unsqueeze(0).unsqueeze(0)
    val dist = torch.clamp((disttotalScores - distcumScores) * positionEffect, min = 0.0f)
    dist.sqrt()
  }

  // 计算总效应
  val softplus = F.softplus(_: Tensor[ParamType])
  val gammaValue = gamma.map(g => -1.0f * softplus(g).unsqueeze(0)).getOrElse(torch.ones(1))

  val totalEffect = if (pdiff.isEmpty) {
    torch.clamp(torch.clamp((distScores * gammaValue).exp, min = 1e-5f), max = 1e5f)
  } else {
    val diff = pdiff.get.unsqueeze(1).expand(pdiff.get.shape(0), distScores.shape(1), pdiff.get.shape(1), pdiff.get.shape(2))
    val diffExp = diff.sigmoid.exp
    torch.clamp(torch.clamp((distScores * gammaValue * diffExp).exp, min = 1e-5f), max = 1e5f)
  }

  // 应用效应并计算最终注意力
  val finalScores = scores * totalEffect
  finalScores.masked_fill_(mask == torch.zeros(1), -1e32f)
  val attentionWeights = F.softmax(finalScores, dim = -1)

  // 零填充（如果需要）
  val paddedWeights = if (zero_pad) {
    val padZero = torch.zeros(bs, head, 1, seqlen).to(device)
    torch.cat(Seq(padZero, attentionWeights.slice(2, 1, -1)), dim = 2)
  } else {
    attentionWeights
  }

  // 应用dropout并计算输出
  val droppedWeights = dropout(paddedWeights)
  torch.matmul(droppedWeights, v)
}

// TransformerLayer类的实现
class TransformerLayer[ParamType <: FloatNN: Default](
    embedding_size: Int,
    d_feature: Int,
    d_ff: Int,
    n_heads: Int,
    dropout: Double,
    kq_same: Boolean
) extends TensorModule[ParamType] with HasParams[ParamType]:
  // 多头注意力块
  val masked_attn_head: MultiHeadAttention[ParamType] = MultiHeadAttention(
    embedding_size, d_feature, n_heads, dropout, kq_same
  )

  // 两个层标准化层和两个dropout层
  val layer_norm1: LayerNorm[ParamType] = LayerNorm(Seq(embedding_size))
  val dropout1: Dropout[ParamType] = Dropout(dropout)

  val linear1: Linear[ParamType] = Linear(embedding_size, d_ff)
  val activation: ReLU[ParamType] = ReLU()
  val dropout_layer: Dropout[ParamType] = Dropout(dropout)
  val linear2: Linear[ParamType] = Linear(d_ff, embedding_size)

  val layer_norm2: LayerNorm[ParamType] = LayerNorm(Seq(embedding_size))
  val dropout2: Dropout[ParamType] = Dropout(dropout)

  def forward(mask: Int, query: Tensor[ParamType], key: Tensor[ParamType], values: Tensor[ParamType], apply_pos: Boolean = true, pdiff: Option[Tensor[ParamType]] = None): Tensor[ParamType] = {
    val seqlen = query.size(1)
    val batch_size = query.size(0)
    val device = query.device
    
    // 创建掩码
    val nopeek_mask = torch.triu(torch.ones(1, 1, seqlen, seqlen), diagonal = mask).to(device)
    val src_mask = (nopeek_mask == torch.zeros(1))

    // 调用多头注意力
    val query2 = if (mask == 0) {
      masked_attn_head(query, key, values, src_mask, zero_pad = true, pdiff)
    } else {
      masked_attn_head(query, key, values, src_mask, zero_pad = false, pdiff)
    }

    // 残差连接和层标准化
    var updatedQuery = query + dropout1(query2)
    updatedQuery = layer_norm1(updatedQuery)

    // 位置前馈网络（如果需要）
    if (apply_pos) {
      val ffnOutput = linear2(dropout_layer(activation(linear1(updatedQuery))))
      updatedQuery = updatedQuery + dropout2(ffnOutput)
      updatedQuery = layer_norm2(updatedQuery)
    }

    updatedQuery
  }

  // apply方法调用forward
  def apply(mask: Int, query: Tensor[ParamType], key: Tensor[ParamType], values: Tensor[ParamType], apply_pos: Boolean = true, pdiff: Option[Tensor[ParamType]] = None): Tensor[ParamType] = {
    forward(mask, query, key, values, apply_pos, pdiff)
  }

// Architecture类的实现
class Architecture[ParamType <: FloatNN: Default](
    num_skills: Int,
    num_blocks: Int,
    embedding_size: Int,
    d_feature: Double,
    d_ff: Int,
    n_heads: Int,
    dropout: Double,
    kq_same: Boolean
) extends TensorModule[ParamType] with HasParams[ParamType]:
  val blocks_1: ListBuffer[TransformerLayer[ParamType]] = ListBuffer()
  val blocks_2: ListBuffer[TransformerLayer[ParamType]] = ListBuffer()

  // 初始化block列表
  for (_ <- 0 until num_blocks) {
    blocks_1 += TransformerLayer(
      embedding_size, embedding_size / n_heads, d_ff, n_heads, dropout, kq_same
    )
  }

  for (_ <- 0 until num_blocks * 2) {
    blocks_2 += TransformerLayer(
      embedding_size, embedding_size / n_heads, d_ff, n_heads, dropout, kq_same
    )
  }

  def forward(q_embed_data: Tensor[ParamType], qa_embed_data: Tensor[ParamType], pid_embed_data: Option[Tensor[ParamType]]): Tensor[ParamType] = {
    // qa和q的位置嵌入
    var y = qa_embed_data
    var x = q_embed_data

    // encoder - 编码qa序列
    for (block <- blocks_1) {
      y = block(mask = 1, query = y, key = y, values = y, pdiff = pid_embed_data)
    }

    var flag_first = true
    for (block <- blocks_2) {
      if (flag_first) {
        // 关注当前问题
        x = block(mask = 1, query = x, key = x, values = x, apply_pos = false, pdiff = pid_embed_data)
        flag_first = false
      } else {
        // 不关注当前回答
        x = block(mask = 0, query = x, key = x, values = y, apply_pos = true, pdiff = pid_embed_data)
        flag_first = true
      }
    }

    x
  }

  // apply方法调用forward
  def apply(q_embed_data: Tensor[ParamType], qa_embed_data: Tensor[ParamType], pid_embed_data: Option[Tensor[ParamType]]): Tensor[ParamType] = {
    forward(q_embed_data, qa_embed_data, pid_embed_data)
  }

// LearnablePositionalEmbedding类的实现
class LearnablePositionalEmbedding[ParamType <: FloatNN: Default](
    embedding_size: Int,
    max_len: Int = 512
) extends TensorModule[ParamType] with HasParams[ParamType]:
  // 初始化可学习的位置编码
  val weight: Tensor[ParamType] = (0.1f * Tensor.randn[ParamType](max_len, embedding_size)).unsqueeze(0).requiresGrad_(true)

  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    weight.slice(1, 0, x.size(Dim.seq.idx))
  }

  // apply方法调用forward
  def apply(x: Tensor[ParamType]): Tensor[ParamType] = {
    forward(x)
  }

// CosinePositionalEmbedding类的实现
class CosinePositionalEmbedding[ParamType <: FloatNN: Default](
    embedding_size: Int,
    max_len: Int = 512
) extends TensorModule[ParamType] with HasParams[ParamType]:
  // 计算位置编码
  val weight: Tensor[ParamType] = withNoGrad {
    val pe = 0.1f * Tensor.randn[ParamType](max_len, embedding_size)
    val position = torch.arange(0, max_len).unsqueeze(1)
    val divTerm = torch.exp(
      torch.arange(0, embedding_size, 2) *
      (-math.log(10000.0) / embedding_size)
    )
    pe.slice(1, 0, -1, 2) := torch.sin(position * divTerm)
    pe.slice(1, 1, -1, 2) := torch.cos(position * divTerm)
    pe.unsqueeze(0)
  }

  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    weight.slice(1, 0, x.size(Dim.seq.idx))
  }

  // apply方法调用forward
  def apply(x: Tensor[ParamType]): Tensor[ParamType] = {
    forward(x)
  }

// AKT主模型类的实现
class AKT[ParamType <: FloatNN: Default](
    joint: Boolean,
    mask_response: Boolean,
    pred_last: Boolean,
    mask_future: Boolean,
    length: Int,
    trans: Boolean,
    num_skills: Int,
    num_questions: Int,
    embedding_size: Int,
    num_blocks: Int,
    dropout: Double,
    kq_same: Boolean,
    d_ff: Int = 256,
    final_fc_dim: Int = 512,
    num_attn_heads: Int = 8,
    separate_qr: Boolean = false,
    l2: Double = 1e-5
) extends TensorModule[ParamType] with HasParams[ParamType]:
  val embed_l: Int = embedding_size

  // 问题难度参数（如果有问题ID）
  val difficult_param: Option[Embedding[ParamType]] = if (num_questions > 0) Some(Embedding(num_questions + 1, 1)) else None
  val q_embed_diff: Option[Embedding[ParamType]] = if (num_questions > 0) Some(Embedding(num_skills + 1, embed_l)) else None
  val qa_embed_diff: Option[Embedding[ParamType]] = if (num_questions > 0) Some(Embedding(2 * num_skills + 1, embed_l)) else None

  // 问题嵌入
  val q_embed: Embedding[ParamType] = Embedding(num_skills, embed_l)
  // 交互嵌入
  val qa_embed: Embedding[ParamType] = if (separate_qr) {
    Embedding(2 * num_skills + 1, embed_l)
  } else {
    Embedding(2, embed_l)
  }

  // Architecture对象，包含注意力块堆栈
  val model: Architecture[ParamType] = Architecture(
    num_skills = num_skills,
    num_blocks = num_blocks,
    n_heads = num_attn_heads,
    dropout = dropout,
    embedding_size = embedding_size,
    d_feature = embedding_size / num_attn_heads,
    d_ff = d_ff,
    kq_same = kq_same
  )

  // 输出层
  val out: Sequential[ParamType] = if (trans) {
    Sequential(
      Linear(embedding_size + embed_l, final_fc_dim),
      ReLU(),
      Dropout(dropout),
      Linear(final_fc_dim, 256),
      ReLU(),
      Dropout(dropout),
      Linear(256, num_skills)
    )
  } else {
    Sequential(
      Linear(embedding_size + embed_l, final_fc_dim),
      ReLU(),
      Dropout(dropout),
      Linear(final_fc_dim, 256),
      ReLU(),
      Dropout(dropout),
      Linear(256, 1)
    )
  }

  // 损失函数
  val loss_fn: BCELoss = BCELoss(reduction = "mean")

  // 初始化
  reset()

  def reset(): Unit = {
    for (p <- this.parameters) {
      if (num_questions > 0 && p.size(0) == num_questions + 1) {
        torch.ops.nn.init.constant_(p, 0.0)
      }
    }
  }

  def base_emb(q_data: Tensor[ParamType], target: Tensor[ParamType]): (Tensor[ParamType], Tensor[ParamType]) = {
    val q_embed_data = q_embed(q_data)
    val qa_embed_data = if (separate_qr) {
      val qa_data = q_data + Tensor(num_skills.toFloat) * target
      qa_embed(qa_data)
    } else {
      qa_embed(target) + q_embed_data
    }
    (q_embed_data, qa_embed_data)
  }

  def mask_future_length(input: Tensor[ParamType], mask_length: Int): Tensor[ParamType] = {
    val last_ones = (input == torch.ones(1)).cumsum(dim = 1).argmax(dim = 1)
    val col_indices = torch.arange(input.shape(1)).unsqueeze(0)
    val mask = col_indices < (last_ones - mask_length + 1).unsqueeze(1)
    val insufficient_ones = last_ones < (mask_length - 1)
    mask(insufficient_ones) := false
    input * mask
  }

  def forward(feed_dict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = {
    val pid_data = feed_dict("questions")
    val r = feed_dict("responses")
    val c = feed_dict("skills")
    val attention_mask = feed_dict("attention_mask")

    var q_data = c
    var target = r * (r > Tensor(-1.0f))

    var cshft: Option[Tensor[ParamType]] = None
    var rshft: Option[Tensor[ParamType]] = None

    // 根据不同模式处理数据
    if (trans) {
      q_data = q_data.slice(1, 0, -length)
      cshft = Some(c.slice(1, length, -1))
      target = target.slice(1, 0, -length)
    } else if (mask_future) {
      // 应用未来掩码
      val maskPart = attention_mask.slice(1, -length, -1).zeros_like()
      val maskedAttention = torch.cat(
        Seq(attention_mask.slice(1, 0, -length), maskPart),
        dim = 1
      )
      q_data = q_data * maskedAttention
      target = target * maskedAttention
    } else if (mask_response) {
      // 应用响应掩码
      val maskPart = attention_mask.slice(1, -length, -1).zeros_like()
      val maskedAttention = torch.cat(
        Seq(attention_mask.slice(1, 0, -length), maskPart),
        dim = 1
      )
      target = target * maskedAttention
    }

    // 获取基础嵌入
    val (q_embed_data, qa_embed_data) = base_emb(q_data, target)

    var c_reg_loss: Tensor[ParamType] = torch.zeros(1)
    var pid_embed_data: Option[Tensor[ParamType]] = None

    // 如果有问题ID，应用难度参数
    if (num_questions > 0) {
      pid_embed_data = difficult_param.map(_(pid_data))
      val q_embed_diff_data = q_embed_diff.get(q_data)
      val updated_q_embed = q_embed_data + pid_embed_data.get * q_embed_diff_data

      val qa_embed_diff_data = qa_embed_diff.get(target)
      val updated_qa_embed = if (separate_qr) {
        qa_embed_data + pid_embed_data.get * qa_embed_diff_data
      } else {
        qa_embed_data + pid_embed_data.get * (qa_embed_diff_data + q_embed_diff_data)
      }

      // 计算正则化损失
      c_reg_loss = (pid_embed_data.get ** 2.0f).sum() * l2
    }

    // 通过解码器
    val d_output = model(q_embed_data, qa_embed_data, pid_embed_data)

    // 计算池化分数
    val masked_q_embed = q_embed(q_data) * attention_mask.unsqueeze(-1)
    val pooled_ques_score = masked_q_embed.sum(1) / attention_mask.sum(-1).unsqueeze(-1)
    val masked_inter = qa_embed_data * attention_mask.unsqueeze(-1)
    val pooled_inter_score = masked_inter.sum(1) / attention_mask.sum(-1).unsqueeze(-1)

    // 连接并通过输出层
    val concat_q = torch.cat(Seq(d_output, q_embed_data), dim = -1)
    var output = if (trans) {
      out(concat_q)
    } else {
      out(concat_q).squeeze(-1)
    }

    // 应用sigmoid激活
    val sigmoid = Sigmoid[ParamType]()
    output = sigmoid(output)

    var true_output: Tensor[ParamType] = torch.zeros(1)

    // 根据不同模式处理输出
    if (trans && cshft.isDefined) {
      if (joint) {
        val seqLen = output.size(1)
        val mid = seqLen / 2
        // 扩展中间部分的输出
        val expandedOutput = output.slice(1, mid, mid + 1).expand(-1, seqLen - mid, -1)
        output = torch.cat(Seq(output.slice(1, 0, mid), expandedOutput), dim = 1)
        
        // 使用one-hot编码选择技能对应的输出
        val oneHot = F.one_hot(cshft.get.toLong, num_skills)
        output = (output * oneHot).sum(-1)
        
        // 选择后半部分输出
        output = output.slice(1, mid, -1)
        true_output = r.slice(1, length + mid, -1)
      } else {
        // 使用one-hot编码选择技能对应的输出
        val oneHot = F.one_hot(cshft.get.toLong, num_skills)
        output = (output * oneHot).sum(-1)
        true_output = r.slice(1, length, -1)
      }
    } else if (mask_future || pred_last || mask_response) {
      output = output.slice(1, -length, -1)
      true_output = r.slice(1, -length, -1)
    } else {
      output = output.slice(1, length, -1)
      true_output = r.slice(1, length, -1)
    }

    // 返回结果
    if (this.isTraining) {
      Map(
        "pred" -> output,
        "true" -> true_output,
        "c_reg_loss" -> c_reg_loss
      )
    } else {
      Map(
        "pred" -> output,
        "true" -> true_output,
        "c_reg_loss" -> c_reg_loss,
        "q_embed" -> pooled_ques_score,
        "qr_embed" -> pooled_inter_score
      )
    }
  }

  // apply方法调用forward
  def apply(feed_dict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = {
    forward(feed_dict)
  }

  // 损失计算方法
  def loss(feed_dict: Map[String, Tensor[ParamType]], out_dict: Map[String, Tensor[ParamType]]): (Tensor[ParamType], Int, Double) = {
    val pred = out_dict("pred").flatten()
    val true_values = out_dict("true").flatten()
    val c_reg_loss = out_dict("c_reg_loss")
    val mask = true_values > Tensor(-1.0f)
    
    // 计算损失
    val loss = loss_fn(pred.masked_select(mask), true_values.masked_select(mask)) + c_reg_loss
    
    // 返回损失值、掩码数量和真实值总和
    (loss, pred.masked_select(mask).numel, true_values.masked_select(mask).sum().double())
  }

// 伴生对象，提供工厂方法
object AKT {
  def apply[ParamType <: FloatNN: Default](
    joint: Boolean,
    mask_response: Boolean,
    pred_last: Boolean,
    mask_future: Boolean,
    length: Int,
    trans: Boolean,
    num_skills: Int,
    num_questions: Int,
    embedding_size: Int,
    num_blocks: Int,
    dropout: Double,
    kq_same: Boolean,
    d_ff: Int = 256,
    final_fc_dim: Int = 512,
    num_attn_heads: Int = 8,
    separate_qr: Boolean = false,
    l2: Double = 1e-5
  ): AKT[ParamType] = {
    new AKT(
      joint, mask_response, pred_last, mask_future, length, trans,
      num_skills, num_questions, embedding_size, num_blocks, dropout, kq_same,
      d_ff, final_fc_dim, num_attn_heads, separate_qr, l2
    )
  }
}
