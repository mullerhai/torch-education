package torch.edu

import torch.*
import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn as nn
import torch.nn.*
import scala.collection.mutable.ListBuffer

// 维度枚举类，用于表示张量的维度
object Dim {
  val batch = 0
  val seq = 1
  val feature = 2
}

// 余弦位置编码
class CosinePositionalEmbedding5[ParamType <: FloatNN: Default](
    embedding_size: Int,
    max_len: Int = 512
) extends HasParams[ParamType]:
  // 计算位置编码并保存为参数
  private val pe = {
    val tempPe = 0.1f * torch.randn[ParamType](Seq(max_len, embedding_size))
    val position = torch.arange(0, max_len).unsqueeze(1)
    val divTerm = torch.exp(
      torch.arange(0, embedding_size, 2) *
      -(math.log(10000.0) / embedding_size)
    )
    tempPe.slice(1, 0, tempPe.shape(0), step = 2).assign(torch.sin(position * divTerm))
    tempPe.slice(1, 1, tempPe.shape(0), step = 2).assign(torch.cos(position * divTerm))
    tempPe.unsqueeze(0)
  }
  
  // 创建不可训练的参数
  private val weight = Tensor[ParamType](pe.data, requires_grad = false)
  
  // 收集参数
  override def params: Seq[Tensor[ParamType]] = Seq(weight)
  
  // 前向传播
  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    weight.slice(1, 0, x.shape(Dim.seq))
  }
  
  // 实现apply方法
  def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)

// 可学习的位置编码
class LearnablePositionalEmbedding7[ParamType <: FloatNN: Default](
    embedding_size: Int,
    max_len: Int = 512
) extends HasParams[ParamType]:
  // 创建可学习的位置编码参数
  private val weight = Parameter[ParamType](
    0.1f * torch.randn[ParamType](Seq(1, max_len, embedding_size))
  )
  
  // 收集参数
  override def params: Seq[Tensor[ParamType]] = Seq(weight)
  
  // 前向传播
  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    weight.slice(1, 0, x.shape(Dim.seq))
  }
  
  // 实现apply方法
  def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)

// 注意力计算函数
def attention7[ParamType <: FloatNN: Default](
    q: Tensor[ParamType],
    k: Tensor[ParamType],
    v: Tensor[ParamType],
    d_k: Int,
    mask: Tensor[Boolean],
    dropout: Dropout[ParamType],
    zero_pad: Boolean
): Tensor[ParamType] = {
  // 计算注意力分数
  val scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
  val bs = scores.shape(0)
  val head = scores.shape(1)
  val seqlen = scores.shape(2)
  val device = q.device
  
  // 应用掩码
  scores.masked_fill_(mask.logicalNot(), -1e32f)
  
  // 应用softmax
  val softmaxScores = F.softmax(scores, dim = -1)
  
  // 应用零填充（如果需要）
  val paddedScores = if (zero_pad) {
    val padZero = torch.zeros(Seq(bs, head, 1, seqlen), device = device)
    torch.cat(Seq(padZero, softmaxScores.slice(2, 1, softmaxScores.shape(2))), dim = 2)
  } else {
    softmaxScores
  }
  
  // 应用dropout并计算最终输出
  val droppedScores = dropout(paddedScores)
  torch.matmul(droppedScores, v)
}

// 多头注意力层
class MultiHeadAttention5[ParamType <: FloatNN: Default](
    embedding_size: Int,
    d_feature: Int,
    n_heads: Int,
    dropout: Double,
    kq_same: Boolean,
    bias: Boolean = true
) extends HasParams[ParamType]:
  // 保存参数
  val d_k = d_feature
  val h = n_heads
  val kq_same_flag = kq_same
  
  // 创建线性层
  private val v_linear = Linear[ParamType](embedding_size, embedding_size, bias = bias)
  private val k_linear = Linear[ParamType](embedding_size, embedding_size, bias = bias)
  private val q_linear = if (!kq_same) Some(Linear[ParamType](embedding_size, embedding_size, bias = bias)) else None
  private val dropoutLayer = Dropout[ParamType](dropout)
  private val out_proj = Linear[ParamType](embedding_size, embedding_size, bias = bias)
  
  // 初始化参数
  _reset_parameters()
  
  // 参数初始化方法
  private def _reset_parameters(): Unit = {
    // Xavier均匀初始化权重
    torch.nn.init.xavier_uniform_(k_linear.weight)
    torch.nn.init.xavier_uniform_(v_linear.weight)
    q_linear.foreach(torch.nn.init.xavier_uniform_(_.weight))
    
    // 偏置初始化为0
    if (bias) {
      torch.nn.init.constant_(k_linear.bias, 0.0f)
      torch.nn.init.constant_(v_linear.bias, 0.0f)
      q_linear.foreach(torch.nn.init.constant_(_.bias, 0.0f))
      torch.nn.init.constant_(out_proj.bias, 0.0f)
    }
  }
  
  // 收集参数
  override def params: Seq[Tensor[ParamType]] = {
    Seq(k_linear.weight, v_linear.weight, out_proj.weight) ++
    (if (bias) Seq(k_linear.bias, v_linear.bias, out_proj.bias) else Seq.empty) ++
    q_linear.map(q => Seq(q.weight) ++ (if (bias) Seq(q.bias) else Seq.empty)).getOrElse(Seq.empty)
  }
  
  // 前向传播
  def forward(
    q: Tensor[ParamType],
    k: Tensor[ParamType],
    v: Tensor[ParamType],
    mask: Tensor[Boolean],
    zero_pad: Boolean
  ): Tensor[ParamType] = {
    val bs = q.shape(0)
    
    // 线性变换并分拆成多个头
    val kTransformed = k_linear(k).view(Seq(bs, -1, h, d_k))
    val qTransformed = if (!kq_same_flag) {
      q_linear.get(q).view(Seq(bs, -1, h, d_k))
    } else {
      k_linear(q).view(Seq(bs, -1, h, d_k))
    }
    val vTransformed = v_linear(v).view(Seq(bs, -1, h, d_k))
    
    // 转置以获得正确的维度顺序
    val kTransposed = kTransformed.transpose(1, 2)
    val qTransposed = qTransformed.transpose(1, 2)
    val vTransposed = vTransformed.transpose(1, 2)
    
    // 计算注意力
    val scores = attention7(qTransposed, kTransposed, vTransposed, d_k, mask, dropoutLayer, zero_pad)
    
    // 拼接所有头并通过最终线性层
    val concat = scores.transpose(1, 2).contiguous()
      .view(Seq(bs, -1, embedding_size))
    
    out_proj(concat)
  }
  
  // 实现apply方法
  def apply(
    q: Tensor[ParamType],
    k: Tensor[ParamType],
    v: Tensor[ParamType],
    mask: Tensor[Boolean],
    zero_pad: Boolean
  ): Tensor[ParamType] = forward(q, k, v, mask, zero_pad)

// Transformer层
class TransformerLayer5[ParamType <: FloatNN: Default](
    embedding_size: Int,
    d_feature: Int,
    d_ff: Int,
    n_heads: Int,
    dropout: Double,
    kq_same: Boolean
) extends HasParams[ParamType]:
  // 创建多头注意力层
  private val masked_attn_head = MultiHeadAttention[ParamType](
    embedding_size,
    d_feature,
    n_heads,
    dropout,
    kq_same
  )
  
  // 创建层归一化和dropout层
  private val layer_norm1 = LayerNorm[ParamType](Seq(embedding_size))
  private val dropout1 = Dropout[ParamType](dropout)
  
  // 创建前馈网络
  private val linear1 = Linear[ParamType](embedding_size, d_ff)
  private val activation = ReLU()
  private val dropoutFFN = Dropout[ParamType](dropout)
  private val linear2 = Linear[ParamType](d_ff, embedding_size)
  
  // 创建层归一化和dropout层
  private val layer_norm2 = LayerNorm[ParamType](Seq(embedding_size))
  private val dropout2 = Dropout[ParamType](dropout)
  
  // 收集参数
  override def params: Seq[Tensor[ParamType]] = {
    masked_attn_head.params ++
    layer_norm1.params ++
    Seq(linear1.weight, linear1.bias, linear2.weight, linear2.bias) ++
    layer_norm2.params
  }
  
  // 前向传播
  def forward(
    mask: Int,
    query: Tensor[ParamType],
    key: Tensor[ParamType],
    values: Tensor[ParamType],
    apply_pos: Boolean
  ): Tensor[ParamType] = {
    val seqlen = query.shape(1)
    val batch_size = query.shape(0)
    val device = query.device
    
    // 创建掩码
    val nopeek_mask = torch.ones(Seq(1, 1, seqlen, seqlen), device = device)
      .triu(diagonal = mask)
    val src_mask = nopeek_mask
    
    // 计算注意力
    val query2 = if (mask == 0) {
      masked_attn_head(query, key, values, src_mask, zero_pad = true)
    } else {
      masked_attn_head(query, key, values, src_mask, zero_pad = false)
    }
    
    // 残差连接和层归一化
    var updatedQuery = query + dropout1(query2)
    updatedQuery = layer_norm1(updatedQuery)
    
    // 如果需要，应用前馈网络
    if (apply_pos) {
      val ffnOutput = linear2(dropoutFFN(activation(linear1(updatedQuery))))
      updatedQuery = updatedQuery + dropout2(ffnOutput)
      updatedQuery = layer_norm2(updatedQuery)
    }
    
    updatedQuery
  }
  
  // 实现apply方法
  def apply(
    mask: Int,
    query: Tensor[ParamType],
    key: Tensor[ParamType],
    values: Tensor[ParamType],
    apply_pos: Boolean
  ): Tensor[ParamType] = forward(mask, query, key, values, apply_pos)

// 架构类，包含多个Transformer层
class Architecture5[ParamType <: FloatNN: Default](
    num_skills: Int,
    num_blocks: Int,
    embedding_size: Int,
    d_feature: Double,
    d_ff: Int,
    n_heads: Int,
    dropout: Double,
    kq_same: Int,
    seq_len: Int
) extends HasParams[ParamType]:
  // 创建多个Transformer层
  private val blocks_2 = (0 until num_blocks).map(_ => 
    TransformerLayer[ParamType](
      embedding_size,
      (embedding_size / n_heads), // d_feature
      d_ff,
      n_heads,
      dropout,
      kq_same == 1
    )
  ).toList
  
  // 创建余弦位置编码
  private val position_emb = CosinePositionalEmbedding[ParamType](embedding_size, seq_len)
  
  // 收集参数
  override def params: Seq[Tensor[ParamType]] = {
    blocks_2.flatMap(_.params) ++ position_emb.params
  }
  
  // 前向传播
  def forward(
    q_embed_data: Tensor[ParamType],
    qa_embed_data: Tensor[ParamType]
  ): Tensor[ParamType] = {
    // 计算序列长度和批次大小
    val seqlen = q_embed_data.shape(1)
    val batch_size = q_embed_data.shape(0)
    
    // 添加位置编码
    val q_posemb = position_emb(q_embed_data)
    val q_embed_with_pos = q_embed_data + q_posemb
    val qa_posemb = position_emb(qa_embed_data)
    val qa_embed_with_pos = qa_embed_data + qa_posemb
    
    // 设置初始输入
    var y = qa_embed_with_pos
    var x = q_embed_with_pos
    
    // 通过所有Transformer层
    for (block <- blocks_2) {
      x = block(mask = 0, query = x, key = x, values = y, apply_pos = true)
    }
    
    x
  }
  
  // 实现apply方法
  def apply(
    q_embed_data: Tensor[ParamType],
    qa_embed_data: Tensor[ParamType]
  ): Tensor[ParamType] = forward(q_embed_data, qa_embed_data)

// SimpleKT主模型
class SimpleKT[ParamType <: FloatNN: Default](
    val mask_response: Boolean,
    val pred_last: Boolean,
    val mask_future: Boolean,
    val length: Int,
    val trans: Boolean,
    val num_skills: Int,
    val num_questions: Int,
    val seq_len: Int,
    val embedding_size: Int,
    val num_blocks: Int,
    val dropout: Double,
    val kq_same: Int,
    val d_ff: Int = 256,
    val final_fc_dim: Int = 512,
    val final_fc_dim2: Int = 256,
    val num_attn_heads: Int = 8,
    val separate_qr: Boolean = false,
    val l2: Double = 1e-5
) extends HasParams[ParamType] with TensorModule[ParamType]:
  // 嵌入维度
  private val embed_l = embedding_size
  
  // 难度参数和差异嵌入（如果有题目ID）
  private val difficult_param = if (num_questions > 0) Some(Embedding[ParamType](num_questions + 1, embed_l)) else None
  private val q_embed_diff = if (num_questions > 0) Some(Embedding[ParamType](num_skills + 1, embed_l)) else None
  private val qa_embed_diff = if (num_questions > 0) Some(Embedding[ParamType](2 * num_skills + 1, embed_l)) else None
  
  // 问题嵌入和问答嵌入
  private val q_embed = Embedding[ParamType](num_skills, embed_l)
  private val qa_embed = if (separate_qr) {
    Embedding[ParamType](2 * num_skills + 1, embed_l)
  } else {
    Embedding[ParamType](2, embed_l)
  }
  
  // 创建架构对象
  private val model = Architecture[ParamType](
    num_skills,
    num_blocks,
    embedding_size,
    embedding_size / num_attn_heads,
    d_ff,
    num_attn_heads,
    dropout,
    kq_same,
    seq_len
  )
  
  // 创建输出层
  private val out = if (trans) {
    Sequential[ParamType](
      Linear[ParamType](embedding_size + embed_l, final_fc_dim),
      ReLU(),
      Dropout[ParamType](dropout),
      Linear[ParamType](final_fc_dim, final_fc_dim2),
      ReLU(),
      Dropout[ParamType](dropout),
      Linear[ParamType](final_fc_dim2, num_skills)
    )
  } else {
    Sequential[ParamType](
      Linear[ParamType](embedding_size + embed_l, final_fc_dim),
      ReLU(),
      Dropout[ParamType](dropout),
      Linear[ParamType](final_fc_dim, final_fc_dim2),
      ReLU(),
      Dropout[ParamType](dropout),
      Linear[ParamType](final_fc_dim2, 1)
    )
  }
  
  // 创建损失函数
  private val loss_fn = nn.BCELoss()//Binarycross_entropyLoss(reduction = "mean")
  
  // 初始化
  reset()
  
  // 重置参数
  def reset(): Unit = {
    for (p <- params) {
      if (num_questions > 0 && p.shape(0) == num_questions + 1) {
        torch.nn.init.constant_(p, 0.0f)
      }
    }
  }
  
  // 基础嵌入函数
  private def base_emb(
    q_data: Tensor[ParamType],
    target: Tensor[ParamType]
  ): (Tensor[ParamType], Tensor[ParamType]) = {
    // 获取问题嵌入
    val q_embed_data = q_embed(q_data)
    
    // 根据是否分离QR获取问答嵌入
    val qa_embed_data = if (separate_qr) {
      val qa_data = q_data + num_skills * target
      this.qa_embed(qa_data)
    } else {
      this.qa_embed(target) + q_embed_data
    }
    
    (q_embed_data, qa_embed_data)
  }
  
  // 收集参数
  override def params: Seq[Tensor[ParamType]] = {
    Seq(q_embed.weight, qa_embed.weight) ++
    (difficult_param.map(_.weight).toSeq ++
     q_embed_diff.map(_.weight).toSeq ++
     qa_embed_diff.map(_.weight).toSeq) ++
    model.params ++
    out.params
  }
  
  // 前向传播
  def forward(feed_dict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = {
    val q = feed_dict("questions")
    val c = feed_dict("skills")
    val r = feed_dict("responses")
    val attention_mask = feed_dict("attention_mask")
    
    // 截取数据
    val qshft = q.slice(1, length, q.shape(1))
    val cshft = c.slice(1, length, c.shape(1))
    val masked_r = r * (r > -1.0f)
    val rshft = masked_r.slice(1, length, masked_r.shape(1))
    
    // 设置数据
    var pid_data = q
    var q_data = c
    var target = masked_r
    
    // 根据不同模式处理数据
    if (trans) {
      pid_data = pid_data.slice(1, 0, pid_data.shape(1) - length)
      q_data = q_data.slice(1, 0, q_data.shape(1) - length)
      target = target.slice(1, 0, target.shape(1) - length)
    } else if (mask_future) {
      // 创建掩码并应用
      val maskFuture = attention_mask.clone()
      maskFuture.slice(1, maskFuture.shape(1) - length, maskFuture.shape(1)).assign(0.0f)
      pid_data = pid_data * maskFuture
      q_data = q_data * maskFuture
      target = target * maskFuture
    } else if (mask_response) {
      // 创建掩码并应用
      val maskResponse = attention_mask.clone()
      maskResponse.slice(1, maskResponse.shape(1) - length, maskResponse.shape(1)).assign(0.0f)
      target = target * maskResponse
    }
    
    // 获取基础嵌入
    var (q_embed_data, qa_embed_data) = base_emb(q_data, target)
    
    // 如果有题目ID，添加难度信息
    if (num_questions > 0) {
      val q_embed_diff_data = q_embed_diff.get(q_data)
      val pid_embed_data = difficult_param.get(pid_data)
      q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
    }
    
    // 通过模型处理
    val d_output = model(q_embed_data, qa_embed_data)
    
    // 拼接问题嵌入和输出
    val concat_q = torch.cat(Seq(d_output, q_embed_data), dim = -1)
    
    // 计算输出
    val output = if (trans) {
      out(concat_q)
    } else {
      out(concat_q).squeeze(-1)
    }
    
    // 应用sigmoid激活函数
    val m = Sigmoid[ParamType]()
    val preds = m(output)
    
    // 根据不同模式处理预测结果
    val (final_preds, true_values) = if (trans) {
      // 转换模式
      val oneHotCSHft = F.one_hot(cshft.long(), num_skills)
      val sumPreds = (preds * oneHotCSHft).sum(-1)
      val trueVals = r.slice(1, length, r.shape(1))
      (sumPreds, trueVals)
    } else if (mask_future || pred_last || mask_response) {
      // 掩码未来或只预测最后部分
      val slicedPreds = preds.slice(1, preds.shape(1) - length, preds.shape(1))
      val slicedTrue = r.slice(1, r.shape(1) - length, r.shape(1))
      (slicedPreds, slicedTrue)
    } else {
      // 普通模式
      val slicedPreds = preds.slice(1, length, preds.shape(1))
      val slicedTrue = r.slice(1, length, r.shape(1))
      (slicedPreds, slicedTrue)
    }
    
    // 返回结果字典
    Map("pred" -> final_preds, "true" -> true_values)
  }
  
  // 计算损失
  def loss(feed_dict: Map[String, Tensor[ParamType]], out_dict: Map[String, Tensor[ParamType]]): (Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]) = {
    val pred = out_dict("pred").flatten()
    val true_values = out_dict("true").flatten()
    
    // 创建掩码，过滤掉无效值
    val mask = true_values > -1.0f
    
    // 计算损失
    val loss = loss_fn(pred(mask), true_values(mask))
    
    // 返回损失值、有效样本数和正样本数
    (loss, mask.sum(), true_values(mask).sum())
  }
  
  // 实现apply方法
  override def apply(input: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = forward(input)

// SimpleKT模型的工厂方法
object SimpleKT {
  def apply[ParamType <: FloatNN: Default](
    mask_response: Boolean,
    pred_last: Boolean,
    mask_future: Boolean,
    length: Int,
    trans: Boolean,
    num_skills: Int,
    num_questions: Int,
    seq_len: Int,
    embedding_size: Int,
    num_blocks: Int,
    dropout: Double,
    kq_same: Int,
    d_ff: Int = 256,
    final_fc_dim: Int = 512,
    final_fc_dim2: Int = 256,
    num_attn_heads: Int = 8,
    separate_qr: Boolean = false,
    l2: Double = 1e-5
  ): SimpleKT[ParamType] = {
    new SimpleKT[ParamType](
      mask_response,
      pred_last,
      mask_future,
      length,
      trans,
      num_skills,
      num_questions,
      seq_len,
      embedding_size,
      num_blocks,
      dropout,
      kq_same,
      d_ff,
      final_fc_dim,
      final_fc_dim2,
      num_attn_heads,
      separate_qr,
      l2
    )
  }
}
