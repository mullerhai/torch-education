package torch.edu

import torch.*
import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn as nn
import torch.nn.*
import scala.collection.mutable.ListBuffer
import scala.math.sqrt

// 枚举类，定义维度索引
enum Dim {
  case batch, seq, feature
}

// 余弦位置编码
class CosinePositionalEmbedding[ParamType <: FloatNN: Default] private (
    val weight: Tensor[ParamType]
) extends HasParams[ParamType] {
  override def params: Seq[Tensor[ParamType]] = Seq(weight)

  def apply(x: Tensor[ParamType]): Tensor[ParamType] = {
    // (1, seq, feature)
    weight.slice(Seq(None, Some(0), None), Seq(None, Some(x.size(Dim.seq.id)), None))
  }
}

object CosinePositionalEmbedding {
  def apply[ParamType <: FloatNN: Default](embeddingSize: Int, maxLen: Int = 512): CosinePositionalEmbedding[ParamType] = {
    // 初始化位置编码
    val pe = torch.randn[ParamType](Seq(1, maxLen, embeddingSize)) * 0.1
    val position = torch.arange(0, maxLen).unsqueeze(1).toDType[ParamType]
    
    // 计算除数项
    val divTerm = torch.exp(
      torch.arange(0, embeddingSize, 2).toDType[ParamType] * 
      -(math.log(10000.0) / embeddingSize)
    )
    
    // 填充正弦和余弦值
    val sinPart = (position * divTerm).sin()
    val cosPart = (position * divTerm).cos()
    
    // 将正弦和余弦值赋值给pe
    pe.slice(Seq(None, None, Some(0)), Seq(None, None, None, 2)).assign(sinPart)
    pe.slice(Seq(None, None, Some(1)), Seq(None, None, None, 2)).assign(cosPart)
    
    // 创建不可训练的参数
    val param = Tensor.parameter[ParamType](pe, requiresGrad = false)
    new CosinePositionalEmbedding(param)
  }
}

// 可学习的位置编码
class LearnablePositionalEmbedding[ParamType <: FloatNN: Default] private (
    val weight: Tensor[ParamType]
) extends HasParams[ParamType] {
  override def params: Seq[Tensor[ParamType]] = Seq(weight)

  def apply(x: Tensor[ParamType]): Tensor[ParamType] = {
    weight.slice(Seq(None, Some(0), None), Seq(None, Some(x.size(Dim.seq.id)), None))
  }
}

object LearnablePositionalEmbedding {
  def apply[ParamType <: FloatNN: Default](embeddingSize: Int, maxLen: Int = 512): LearnablePositionalEmbedding[ParamType] = {
    val pe = torch.randn[ParamType](Seq(1, maxLen, embeddingSize)) * 0.1
    val param = Tensor.parameter[ParamType](pe)
    new LearnablePositionalEmbedding(param)
  }
}

// 注意力计算函数
private def attention[ParamType <: FloatNN: Default](
    q: Tensor[ParamType], k: Tensor[ParamType], v: Tensor[ParamType], d_k: Int,
    mask: Tensor[ParamType], dropout: Dropout[ParamType], zeroPad: Boolean,
    embType: String = "qid", sparseRatio: Double = 0.8, kIndex: Int = 5
): (Tensor[ParamType], Tensor[ParamType]) = {
  // 计算注意力分数
  val scores = q.matmul(k.transpose(-2, -1)) / sqrt(d_k)
  val bs = scores.size(0)
  val head = scores.size(1)
  val seqlen = scores.size(2)
  
  // 应用掩码
  scores.masked_fill(mask == 0, -1e32f)
  var attnWeights = F.softmax(scores, dim = -1)
  var beforeDropoutScores = attnWeights
  
  // 稀疏注意力机制
  if (embType.contains("sparseattn")) {
    if (kIndex < seqlen) {
      val scoresA = attnWeights.slice(Seq(None, None, Some(0), None), Seq(None, None, Some(kIndex), None))
      val scoresB = attnWeights.slice(Seq(None, None, Some(kIndex), None), Seq())
        .reshape(Seq(bs * head * (seqlen - kIndex), -1))
      
      val (sortedScores, sortedIdx) = scoresB.sort(dim = -1, descending = true)
      val scoresT = sortedScores.slice(Seq(None, Some(kIndex - 1)), Seq(None, Some(kIndex)))
        .repeat(Seq(1, seqlen))
      
      val newScoresB = scoresB.where(
        scoresB - scoresT >= 0, 
        torch.full_like(scoresB, -1e32f)
      ).reshape(Seq(bs, head, seqlen - kIndex, -1))
      
      attnWeights = torch.cat(Seq(scoresA, newScoresB), dim = 2)
      attnWeights = F.softmax(attnWeights, dim = -1)
    }
  } else if (embType.contains("accumulative")) {
    val reshapedScores = attnWeights.reshape(Seq(bs * head * seqlen, -1))
    val (sortedScores, sortedIdx) = reshapedScores.sort(dim = -1, descending = true)
    val accScores = sortedScores.cumsum(dim = 1)
    
    val accScoresA = accScores.where(accScores <= 0.999f, Tensor.zerosLike(accScores))
    val accScoresB = accScores.where(accScores >= sparseRatio, torch.ones_like(accScores)).where(
      accScores < sparseRatio, Tensor.zerosLike(accScores)
    )
    
    val idx = accScoresB.argmax(dim = 1, keepdim = true)
    val newMask = torch.zeros(Seq(bs * head * seqlen, seqlen))
    val a = torch.ones(Seq(bs * head * seqlen, seqlen))
    newMask.scatter_(1, idx, a)
    
    val idxMatrix = torch.arange(0, seqlen).repeat(Seq(bs * seqlen * head, 1)).toDType[ParamType]
    val newMask2 = idxMatrix.where(idxMatrix - idx <= 0, Tensor.zerosLike(idxMatrix))
      .where(idxMatrix - idx > 0, torch.ones_like(idxMatrix))
    
    val maskedSortedScores = newMask2 * sortedScores
    val maskedSortedScores2 = maskedSortedScores.where(
      maskedSortedScores == 0.0f, torch.full_like(maskedSortedScores, -1f)
    )
    
    val (tmpScores, indices) = maskedSortedScores2.max(dim = 1)
    val tmpScoresRepeat = tmpScores.unsqueeze(-1).repeat(Seq(1, seqlen))
    
    val newScores = reshapedScores.where(
      tmpScoresRepeat - reshapedScores >= 0, torch.full_like(reshapedScores, -1e32f)
    ).reshape(Seq(bs, head, seqlen, -1))
    
    attnWeights = F.softmax(newScores, dim = -1)
  }
  
  // 零填充
  if (zeroPad) {
    val padZero = torch.zeros(Seq(bs, head, 1, seqlen))
    attnWeights = torch.cat(
      Seq(padZero, attnWeights.slice(Seq(None, None, Some(1), None), Seq())), 
      dim = 2
    )
  }
  
  // 应用dropout
  attnWeights = dropout(attnWeights)
  
  // 计算输出
  val output = attnWeights.matmul(v)
  
  if (embType != "qid") {
    (output, attnWeights)
  } else {
    (output, beforeDropoutScores)
  }
}

// 多头注意力层
class MultiHeadAttention6[ParamType <: FloatNN: Default](
    embeddingSize: Int,
    dFeature: Int,
    nHeads: Int,
    dropout: Double,
    kqSame: Boolean,
    bias: Boolean = true
) extends HasParams[ParamType] with TensorModule[ParamType] {
  val dK = dFeature
  val h = nHeads
  val vLinear = Linear[ParamType](embeddingSize, embeddingSize, bias = bias)
  val kLinear = Linear[ParamType](embeddingSize, embeddingSize, bias = bias)
  val qLinearOpt = if (!kqSame) Some(Linear[ParamType](embeddingSize, embeddingSize, bias = bias)) else None
  val dropoutLayer = Dropout[ParamType](dropout)
  val outProj = Linear[ParamType](embeddingSize, embeddingSize, bias = bias)
  
  // 初始化参数
  resetParameters()
  
  private def resetParameters(): Unit = {
    // Xavier均匀初始化权重
    kLinear.weight.data.apply(xavierUniformInit)
    vLinear.weight.data.apply(xavierUniformInit)
    qLinearOpt.foreach(_.weight.data.apply(xavierUniformInit))
    outProj.weight.data.apply(xavierUniformInit)
    
    // 偏置初始化为0
    if (bias) {
      kLinear.bias.foreach(_.data.fill_(0f))
      vLinear.bias.foreach(_.data.fill_(0f))
      qLinearOpt.foreach(_.bias.foreach(_.data.fill_(0f)))
      outProj.bias.foreach(_.data.fill_(0f))
    }
  }
  
  // Xavier均匀初始化函数
  private def xavierUniformInit(tensor: Tensor[ParamType]): Unit = {
    val gain = sqrt(2.0)
    val fanIn = tensor.size(-2).toFloat
    val fanOut = tensor.size(-1).toFloat
    val std = gain * sqrt(2.0 / (fanIn + fanOut))
    val bound = sqrt(3.0) * std
    tensor.uniform_(-bound, bound)
  }
  
  override def params: Seq[Tensor[ParamType]] = {
    val paramsList = ListBuffer(vLinear.params, kLinear.params, outProj.params)
    qLinearOpt.foreach(q => paramsList += q.params)
    paramsList.flatten
  }
  
  def forward(
    q: Tensor[ParamType], k: Tensor[ParamType], v: Tensor[ParamType],
    mask: Tensor[ParamType], zeroPad: Boolean,
    embType: String = "qid", sparseRatio: Double = 0.8, kIndex: Int = 5
  ): (Tensor[ParamType], Tensor[ParamType]) = {
    val bs = q.size(0)
    
    // 线性变换并分块
    val kTransformed = kLinear(k).view(Seq(bs, -1, h, dK))
    val qTransformed = qLinearOpt match {
      case Some(qLinear) => qLinear(q).view(Seq(bs, -1, h, dK))
      case None => kLinear(q).view(Seq(bs, -1, h, dK))
    }
    val vTransformed = vLinear(v).view(Seq(bs, -1, h, dK))
    
    // 转置以获得正确的维度顺序
    val kTransposed = kTransformed.transpose(1, 2)
    val qTransposed = qTransformed.transpose(1, 2)
    val vTransposed = vTransformed.transpose(1, 2)
    
    // 计算注意力
    val (scores, attnWeights) = attention(
      qTransposed, kTransposed, vTransposed, dK, mask, dropoutLayer, zeroPad,
      embType, sparseRatio, kIndex
    )
    
    // 拼接多头并通过输出投影
    val concat = scores.transpose(1, 2).contiguous().view(Seq(bs, -1, embeddingSize))
    val output = outProj(concat)
    
    (output, attnWeights)
  }
  
  // 实现apply方法调用forward
  def apply(
    q: Tensor[ParamType], k: Tensor[ParamType], v: Tensor[ParamType],
    mask: Tensor[ParamType], zeroPad: Boolean,
    embType: String = "qid", sparseRatio: Double = 0.8, kIndex: Int = 5
  ): (Tensor[ParamType], Tensor[ParamType]) = {
    forward(q, k, v, mask, zeroPad, embType, sparseRatio, kIndex)
  }
}

// Transformer层
class TransformerLayer6[ParamType <: FloatNN: Default](
    embeddingSize: Int,
    dFeature: Int,
    dFF: Int,
    nHeads: Int,
    dropout: Double,
    kqSame: Boolean
) extends HasParams[ParamType] with TensorModule[ParamType] {
  val maskedAttnHead = MultiHeadAttention(
    embeddingSize, dFeature, nHeads, dropout, kqSame
  )
  val layerNorm1 = LayerNorm[ParamType](embeddingSize)
  val dropout1 = Dropout[ParamType](dropout)
  
  val linear1 = Linear[ParamType](embeddingSize, dFF)
  val activation = ReLU[ParamType]()
  val dropout = Dropout[ParamType](dropout)
  val linear2 = Linear[ParamType](dFF, embeddingSize)
  
  val layerNorm2 = LayerNorm[ParamType](embeddingSize)
  val dropout2 = Dropout[ParamType](dropout)
  
  override def params: Seq[Tensor[ParamType]] = {
    maskedAttnHead.params ++ 
    layerNorm1.params ++ 
    Seq(dropout1) ++ 
    linear1.params ++ 
    activation.params ++ 
    Seq(dropout) ++ 
    linear2.params ++ 
    layerNorm2.params ++ 
    Seq(dropout2)
  }
  
  def forward(
    mask: Int, query: Tensor[ParamType], key: Tensor[ParamType], values: Tensor[ParamType],
    applyPos: Boolean = true, embType: String = "qid", sparseRatio: Double = 0.8, kIndex: Int = 5
  ): (Tensor[ParamType], Tensor[ParamType]) = {
    val seqlen = query.size(Dim.seq.id)
    val batchSize = query.size(Dim.batch.id)
    
    // 创建掩码
    val nopeekMask = torch.ones(Seq(1, 1, seqlen, seqlen))
      .triu(k = mask)
      .toDType[IntNN] == 0
    val srcMask = nopeekMask.toDType[ParamType]
    
    // 注意力计算
    val (query2, attnWeights) = if (mask == 0) {
      maskedAttnHead(query, key, values, srcMask, zeroPad = true, embType, sparseRatio, kIndex)
    } else {
      maskedAttnHead(query, key, values, srcMask, zeroPad = false, embType, sparseRatio, kIndex)
    }
    
    // 残差连接和层归一化
    val queryUpdated = query + dropout1(query2)
    val queryNorm = layerNorm1(queryUpdated)
    
    // 前馈网络
    if (applyPos) {
      val query3 = linear2(dropout(activation(linear1(queryNorm))))
      val queryWithFFN = queryNorm + dropout2(query3)
      val output = layerNorm2(queryWithFFN)
      (output, attnWeights)
    } else {
      (queryNorm, attnWeights)
    }
  }
  
  // 实现apply方法调用forward
  def apply(
    mask: Int, query: Tensor[ParamType], key: Tensor[ParamType], values: Tensor[ParamType],
    applyPos: Boolean = true, embType: String = "qid", sparseRatio: Double = 0.8, kIndex: Int = 5
  ): (Tensor[ParamType], Tensor[ParamType]) = {
    forward(mask, query, key, values, applyPos, embType, sparseRatio, kIndex)
  }
}

// 架构类
class Architecture6[ParamType <: FloatNN: Default](
    numSkills: Int,
    numBlocks: Int,
    embeddingSize: Int,
    dFeature: Int,
    dFF: Int,
    nHeads: Int,
    dropout: Double,
    kqSame: Boolean,
    modelType: String,
    seqLen: Int
) extends HasParams[ParamType] with TensorModule[ParamType] {
  // 创建Transformer层列表
  val blocks2 = (0 until numBlocks).map {
    _ => TransformerLayer(
      embeddingSize, dFeature, dFF, nHeads, dropout, kqSame
    )
  }
  
  // 位置编码
  val positionEmb = CosinePositionalEmbedding[ParamType](embeddingSize, seqLen)
  
  override def params: Seq[Tensor[ParamType]] = {
    blocks2.flatMap(_.params) ++ positionEmb.params
  }
  
  def forward(
    qEmbedData: Tensor[ParamType], qaEmbedData: Tensor[ParamType],
    embType: String = "qid", sparseRatio: Double = 0.8, kIndex: Int = 5
  ): (Tensor[ParamType], Tensor[ParamType]) = {
    val seqlen = qEmbedData.size(Dim.seq.id)
    
    // 添加位置编码
    val qPosEmb = positionEmb(qEmbedData)
    val qEmbedWithPos = qEmbedData + qPosEmb
    val qaPosEmb = positionEmb(qaEmbedData)
    val qaEmbedWithPos = qaEmbedData + qaPosEmb
    
    var x = qEmbedWithPos
    var y = qaEmbedWithPos
    var attnWeights: Tensor[ParamType] = null
    
    // 通过所有Transformer层
    for (block <- blocks2) {
      val (output, weights) = block(
        mask = 0, query = x, key = x, values = y,
        applyPos = true, embType = embType,
        sparseRatio = sparseRatio, kIndex = kIndex
      )
      x = output
      attnWeights = weights
    }
    
    (x, attnWeights)
  }
  
  // 实现apply方法调用forward
  def apply(
    qEmbedData: Tensor[ParamType], qaEmbedData: Tensor[ParamType],
    embType: String = "qid", sparseRatio: Double = 0.8, kIndex: Int = 5
  ): (Tensor[ParamType], Tensor[ParamType]) = {
    forward(qEmbedData, qaEmbedData, embType, sparseRatio, kIndex)
  }
}

// SparseKT主模型
class sparseKT[ParamType <: FloatNN: Default](
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
    val kq_same: Boolean,
    val d_ff: Int = 256,
    val final_fc_dim: Int = 512,
    val final_fc_dim2: Int = 256,
    val num_attn_heads: Int = 8,
    val separate_qr: Boolean = false,
    val emb_type: String = "qid_sparseattn",
    val sparse_ratio: Double = 0.8,
    val k_index: Int = 5
) extends HasParams[ParamType] with TensorModule[ParamType] {
  val model_name = "sparsekt"
  val model_type = model_name
  val embed_l = embedding_size
  
  // 嵌入层参数
  val difficult_param_opt = if (num_questions > 0) {
    if (emb_type.contains("scalar")) {
      Some(Embedding[ParamType](num_questions + 1, 1))
    } else {
      Some(Embedding[ParamType](num_questions + 1, embed_l))
    }
  } else None
  
  val q_embed_diff_opt = if (num_questions > 0) {
    Some(Embedding[ParamType](num_skills + 1, embed_l))
  } else None
  
  val qa_embed_diff_opt = if (num_questions > 0) {
    Some(Embedding[ParamType](2 * num_skills + 1, embed_l))
  } else None
  
  val q_embed_opt = if (emb_type.startsWith("qid")) {
    Some(Embedding[ParamType](num_skills, embed_l))
  } else None
  
  val qa_embed_opt = if (emb_type.startsWith("qid")) {
    if (separate_qr) {
      Some(Embedding[ParamType](2 * num_skills + 1, embed_l))
    } else {
      Some(Embedding[ParamType](2, embed_l))
    }
  } else None
  
  // 架构模型
  val model = Architecture[
    ParamType
  ](
    num_skills = num_skills,
    num_blocks = num_blocks,
    n_heads = num_attn_heads,
    dropout = dropout,
    embedding_size = embedding_size,
    d_feature = embedding_size / num_attn_heads,
    d_ff = d_ff,
    kq_same = kq_same,
    model_type = model_type,
    seq_len = seq_len
  )
  
  // 输出层
  val out = if (trans) {
    Sequential[
      ParamType
    ](
      Linear(embedding_size + embed_l, final_fc_dim),
      ReLU(),
      Dropout(dropout),
      Linear(final_fc_dim, final_fc_dim2),
      ReLU(),
      Dropout(dropout),
      Linear(final_fc_dim2, num_skills)
    )
  } else {
    Sequential[
      ParamType
    ](
      nn.Linear(embedding_size + embed_l, final_fc_dim),
      ReLU(),
      Dropout(dropout),
      Linear(final_fc_dim, final_fc_dim2),
      ReLU(),
      Dropout(dropout),
      Linear(final_fc_dim2, 1)
    )
  }
  
  // 损失函数
  val loss_fn = BCELoss(reduction = Reduction.Mean)
  
  // 初始化参数
  reset()
  
  def reset(): Unit = {
    if (num_questions > 0) {
      difficult_param_opt.foreach { param =>
        param.weight.data.fill_(0f)
      }
    }
  }
  
  def base_emb(q_data: Tensor[ParamType], target: Tensor[ParamType]): (Tensor[ParamType], Tensor[ParamType]) = {
    val q_embed_data = q_embed_opt.get(q_data)
    val qa_embed_data = if (separate_qr) {
      val qa_data = q_data + num_skills * target.toDType[IntNN]
      qa_embed_opt.get(qa_data)
    } else {
      qa_embed_opt.get(target.toDType[IntNN]) + q_embed_data
    }
    (q_embed_data, qa_embed_data)
  }
  
  override def params: Seq[Tensor[ParamType]] = {
    val paramsList = ListBuffer(model.params, out.params)
    difficult_param_opt.foreach(p => paramsList += p.params)
    q_embed_diff_opt.foreach(p => paramsList += p.params)
    qa_embed_diff_opt.foreach(p => paramsList += p.params)
    q_embed_opt.foreach(p => paramsList += p.params)
    qa_embed_opt.foreach(p => paramsList += p.params)
    paramsList.flatten
  }
  
  var attn_weights: Tensor[ParamType] = _
  
  def forward(feed_dict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = {
    val pid_data = feed_dict("questions")
    val r = feed_dict("responses")
    val c = feed_dict("skills")
    val attention_mask = feed_dict("attention_mask")
    var q_data = c
    var target = r * (r > -1).toDType[IntNN]
    
    var cshft: Tensor[ParamType] = null
    var true_values: Tensor[ParamType] = null
    
    // 根据模式处理输入
    if (trans) {
      pid_data = pid_data.slice(Seq(None, None), Seq(None, Some(-length)))
      q_data = q_data.slice(Seq(None, None), Seq(None, Some(-length)))
      target = target.slice(Seq(None, None), Seq(None, Some(-length)))
      cshft = c.slice(Seq(None, Some(length)), Seq())
      attention_mask = attention_mask.slice(Seq(None, None), Seq(None, Some(-length)))
    } else if (mask_future) {
      val mask = Tensor.ones_like(attention_mask)
      mask.slice(Seq(None, Some(-length)), Seq()).fill_(0f)
      attention_mask = attention_mask * mask
      pid_data = pid_data * attention_mask
      q_data = q_data * attention_mask
      target = target * attention_mask
    } else if (mask_response) {
      val mask = Tensor.ones_like(attention_mask)
      mask.slice(Seq(None, Some(-length)), Seq()).fill_(0f)
      attention_mask = attention_mask * mask
      target = target * attention_mask
    }
    
    // 获取嵌入
    var q_embed_data: Tensor[ParamType] = null
    var qa_embed_data: Tensor[ParamType] = null
    
    if (emb_type.startsWith("qid")) {
      val (q_emb, qa_emb) = base_emb(q_data, target)
      q_embed_data = q_emb
      qa_embed_data = qa_emb
    }
    
    // 处理问题难度参数
    if (num_questions > 0 && !emb_type.contains("norasch")) {
      if (!emb_type.contains("aktrasch")) {
        val q_embed_diff_data = q_embed_diff_opt.get(q_data)
        val pid_embed_data = difficult_param_opt.get(pid_data)
        q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
      } else {
        val q_embed_diff_data = q_embed_diff_opt.get(q_data)
        val pid_embed_data = difficult_param_opt.get(pid_data)
        q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
        
        val qa_embed_diff_data = qa_embed_diff_opt.get(target.toDType[IntNN])
        qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data + q_embed_diff_data)
      }
    }
    
    // 模型前向传播
    var output: Tensor[ParamType] = null
    var preds: Tensor[ParamType] = null
    
    if (Seq("qid", "qidaktrasch", "qid_scalar", "qid_norasch").contains(emb_type)) {
      val (d_output, attn_weights_val) = model(q_embed_data, qa_embed_data)
      attn_weights = attn_weights_val
      
      val concat_q = torch.cat(Seq(d_output, q_embed_data), dim = -1)
      output = out(concat_q).squeeze(-1)
      preds = torch.sigmoid(output)
    } else if (emb_type.contains("attn")) {
      val (d_output, attn_weights_val) = model(
        q_embed_data, qa_embed_data,
        emb_type = emb_type,
        sparse_ratio = sparse_ratio,
        k_index = k_index
      )
      attn_weights = attn_weights_val
      
      val concat_q = torch.cat(Seq(d_output, q_embed_data), dim = -1)
      if (trans) {
        output = out(concat_q)
      } else {
        output = out(concat_q).squeeze(-1)
      }
      preds = torch.sigmoid(output)
      
      // 根据模式处理预测结果
      if (trans) {
        val oneHot = F.one_hot(cshft.toDType[IntNN], num_skills)
        preds = (preds * oneHot).sum(-1)
        true_values = r.slice(Seq(None, Some(length)), Seq()).toDType[ParamType]
      } else if (mask_future || pred_last || mask_response) {
        preds = preds.slice(Seq(None, Some(-length)), Seq())
        true_values = r.slice(Seq(None, Some(-length)), Seq()).toDType[ParamType]
      } else {
        preds = preds.slice(Seq(None, Some(length)), Seq())
        true_values = r.slice(Seq(None, Some(length)), Seq()).toDType[ParamType]
      }
    }
    
    // 计算池化后的分数
    val pooled_ques_score = (q_embed_opt.get(q_data) * attention_mask.unsqueeze(-1)).sum(1)
      / attention_mask.sum(-1).unsqueeze(-1)
    val pooled_inter_score = (qa_embed_data * attention_mask.unsqueeze(-1)).sum(1)
      / attention_mask.sum(-1).unsqueeze(-1)
    
    // 返回结果
    if (training) {
      Map(
        "pred" -> preds,
        "true" -> true_values
      )
    } else {
      Map(
        "pred" -> preds,
        "true" -> true_values,
        "q_embed" -> pooled_ques_score,
        "qr_embed" -> pooled_inter_score
      )
    }
  }
  
  // 损失计算方法
  def loss(feed_dict: Map[String, Tensor[ParamType]], out_dict: Map[String, Tensor[ParamType]]): (Tensor[ParamType], Int, Double) = {
    val pred = out_dict("pred").flatten()
    val true_val = out_dict("true").flatten()
    val mask = true_val > -1
    val loss_val = loss_fn(pred.masked_select(mask), true_val.masked_select(mask))
    
    // 计算样本数量和正样本数量
    val count = mask.sum().item().int()
    val sumTrue = true_val.masked_select(mask).sum().item()
    
    (loss_val, count, sumTrue)
  }
  
  // 实现apply方法调用forward
  def apply(feed_dict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = {
    forward(feed_dict)
  }
}

// 工厂方法
object sparseKT {
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
    kq_same: Boolean,
    d_ff: Int = 256,
    final_fc_dim: Int = 512,
    final_fc_dim2: Int = 256,
    num_attn_heads: Int = 8,
    separate_qr: Boolean = false,
    emb_type: String = "qid_sparseattn",
    sparse_ratio: Double = 0.8,
    k_index: Int = 5
  ): sparseKT[ParamType] = {
    new sparseKT(
      mask_response, pred_last, mask_future, length, trans, num_skills,
      num_questions, seq_len, embedding_size, num_blocks, dropout, kq_same,
      d_ff, final_fc_dim, final_fc_dim2, num_attn_heads, separate_qr,
      emb_type, sparse_ratio, k_index
    )
  }
}
