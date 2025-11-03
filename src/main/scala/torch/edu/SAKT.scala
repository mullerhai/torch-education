package torch.edu

import torch.*
import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn as nn
import torch.nn.*
import scala.collection.mutable.ListBuffer

// 前馈神经网络组件
class transformer_FFN[ParamType <: FloatNN: Default](
    emb_size: Int,
    dropout: Double
) extends HasParams[ParamType] with TensorModule[ParamType]:
  // 构建前馈网络，包含两个线性层和ReLU激活函数
  private val linear1 = Linear[ParamType](emb_size, emb_size)
  private val linear2 = Linear[ParamType](emb_size, emb_size)
  private val relu = ReLU()
  private val dropoutLayer = Dropout[ParamType](dropout)
  
  // 收集所有参数
  override def params: Seq[Tensor[ParamType]] = Seq(linear1.weight, linear1.bias, linear2.weight, linear2.bias)
  
  // 前向传播
  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
    val x = linear1(input)
    val xRelu = relu(x)
    val xDropout = dropoutLayer(xRelu)
    val output = linear2(xDropout)
    output
  }
  
  // 实现apply方法，调用forward
  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)

// Transformer块组件
class Blocks[ParamType <: FloatNN: Default](
    embedding_size: Int,
    num_attn_heads: Int,
    dropout: Double
) extends HasParams[ParamType] with TensorModule[ParamType]:
  // 多头注意力层
  private val attn = MultiheadAttention[ParamType](
    embedding_size,
    num_attn_heads,
    dropout = dropout
  )
  
  // 前馈网络层
  private val ffn = transformer_FFN[ParamType](embedding_size, dropout)
  
  // 注意力层的dropout和层归一化
  private val attn_dropout = Dropout[ParamType](dropout)
  private val attn_layer_norm = LayerNorm[ParamType](Seq(embedding_size))
  
  // 前馈网络的dropout和层归一化
  private val ffn_dropout = Dropout[ParamType](dropout)
  private val ffn_layer_norm = LayerNorm[ParamType](Seq(embedding_size))
  
  // 收集所有参数
  override def params: Seq[Tensor[ParamType]] = {
    attn.params ++
    ffn.params ++
    attn_layer_norm.params ++
    ffn_layer_norm.params
  }
  
  // 前向传播
  def forward(q: Tensor[ParamType], k: Tensor[ParamType], v: Tensor[ParamType]): Tensor[ParamType] = {
    // 转换维度顺序，适应MultiheadAttention的输入要求
    val qPermuted = q.permute(1, 0, 2)
    val kPermuted = k.permute(1, 0, 2)
    val vPermuted = v.permute(1, 0, 2)
    
    // 获取设备并创建上三角掩码
    val device = q.device
    val causalMask = ut_mask(kPermuted.shape(0), device)
    
    // 多头注意力计算
    val (attnEmb, _) = attn(qPermuted, kPermuted, vPermuted, attn_mask = causalMask)
    
    // 注意力层的dropout和残差连接
    val attnDropout = attn_dropout(attnEmb)
    val attnDropoutPermuted = attnDropout.permute(1, 0, 2)
    val qPermutedBack = qPermuted.permute(1, 0, 2)
    
    // 注意力层的层归一化
    val attnEmbNorm = attn_layer_norm(qPermutedBack + attnDropoutPermuted)
    
    // 前馈网络处理
    val ffnEmb = ffn(attnEmbNorm)
    val ffnDropout = ffn_dropout(ffnEmb)
    
    // 前馈网络的层归一化和残差连接
    val ffnEmbNorm = ffn_layer_norm(attnEmbNorm + ffnDropout)
    
    ffnEmbNorm
  }
  
  // 实现apply方法，调用forward
  override def apply(q: Tensor[ParamType], k: Tensor[ParamType], v: Tensor[ParamType]): Tensor[ParamType] = 
    forward(q, k, v)

// SAKT主模型
class SAKT[ParamType <: FloatNN: Default](
    val joint: Boolean,
    val mask_future: Boolean,
    val length: Int,
    val trans: Boolean,
    val num_skills: Int,
    val seq_len: Int,
    val embedding_size: Int,
    val num_attn_heads: Int,
    val dropout: Double,
    val num_blocks: Int = 2
) extends HasParams[ParamType] with TensorModule[ParamType]:
  // 交互嵌入和练习嵌入
  private val interaction_emb = nn.Embedding[ParamType](num_skills * 2, embedding_size)
  private val exercise_emb = nn.Embedding[ParamType](num_skills, embedding_size)
  private val position_emb = nn.Embedding[ParamType](seq_len, embedding_size)
  
  // 创建多个Transformer块
  private val blocks = get_clones(new Blocks[ParamType](embedding_size, num_attn_heads, dropout), num_blocks)
  
  // Dropout层
  private val dropout_layer = Dropout[ParamType](dropout)
  
  // 预测层
  private val pred = if (trans) Linear[ParamType](embedding_size, num_skills) else Linear[ParamType](embedding_size, 1)
  
  // BCE损失函数
  private val loss_fn = nn.BCELoss() // F.binary_cross_entropy()//.Binarycross_entropyLoss(reduction = "mean")
  
  
  // 收集所有参数
  override def params: Seq[Tensor[ParamType]] = {
    Seq(interaction_emb.weight, exercise_emb.weight, position_emb.weight) ++
    blocks.flatMap(_.params) ++
    Seq(pred.weight, pred.bias)
  }
  
  // 基础嵌入函数
  private def base_emb(
    q: Tensor[ParamType], 
    r: Tensor[ParamType], 
    qry: Tensor[ParamType]
  ): (Tensor[ParamType], Tensor[ParamType]) = {
    // 计算交互索引
    val x = q + num_skills * r
    
    // 获取嵌入
    val qshftemb = exercise_emb(qry)
    val xemb = interaction_emb(x)
    
    // 添加位置编码
    val device = q.device
    val posemb = position_emb(pos_encode(xemb.shape(1), device))
    val xembWithPos = xemb + posemb
    
    (qshftemb, xembWithPos)
  }
  
  // 前向传播
  def forward(feed_dict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = {
    val q = feed_dict("skills")
    val r = feed_dict("responses")
    
    // 处理响应掩码
    val masked_r = r * (r > -1.0f)
    
    // 准备查询数据
    val qry = q.slice(1, length, q.shape(1))
    
    var cshft: Tensor[ParamType] = null
    if (trans) {
      cshft = q.slice(1, length, q.shape(1))
    }
    
    // 截取历史数据
    val qHistorical = q.slice(1, 0, q.shape(1) - length)
    val maskedRHistorical = masked_r.slice(1, 0, masked_r.shape(1) - length)
    
    // 获取基础嵌入
    val (qshftemb, xemb) = base_emb(qHistorical, maskedRHistorical, qry)
    
    // 通过所有Transformer块
    var currentEmb = xemb
    for (block <- blocks) {
      currentEmb = block(qshftemb, currentEmb, currentEmb)
    }
    
    // 根据不同模式计算预测结果
    val (p, trueValues) = if (trans) {
      if (joint) {
        // 联合模式
        val pRaw = pred(dropout_layer(currentEmb))
        val seqLen = pRaw.shape(1)
        val mid = seqLen / 2
        
        // 复制中间部分的预测结果到后半部分
        val expandedPart = pRaw.slice(1, mid, mid + 1).expand(-1, seqLen - mid, -1)
        val pExpanded = pRaw.slice(1, 0, mid).cat(expandedPart, dim = 1)
        
        // 应用sigmoid并与one-hot编码相乘
        val pSigmoid = torch.sigmoid(pExpanded)
        val oneHotCSHft = F.one_hot(cshft.long(), num_skills)
        val pSum = (pSigmoid * oneHotCSHft).sum(-1)
        
        // 截取后半部分的预测结果和真实值
        val pFinal = pSum.slice(1, mid, pSum.shape(1))
        val rshft = r.slice(1, length, r.shape(1))
        val trueFinal = rshft.slice(1, mid, rshft.shape(1))
        
        (pFinal, trueFinal)
      } else {
        // 非联合模式
        val pRaw = torch.sigmoid(pred(dropout_layer(currentEmb)))
        val oneHotCSHft = F.one_hot(cshft.long(), num_skills)
        val pSum = (pRaw * oneHotCSHft).sum(-1)
        val trueFinal = r.slice(1, length, r.shape(1))
        
        (pSum, trueFinal)
      }
    } else if (mask_future) {
      // 掩码未来模式
      val pRaw = torch.sigmoid(pred(dropout_layer(currentEmb))).squeeze(-1)
      val pFinal = pRaw.slice(1, pRaw.shape(1) - length, pRaw.shape(1))
      val trueFinal = r.slice(1, r.shape(1) - length, r.shape(1))
      
      (pFinal, trueFinal)
    } else {
      // 普通模式
      val pRaw = torch.sigmoid(pred(dropout_layer(currentEmb))).squeeze(-1)
      val trueFinal = r.slice(1, length, r.shape(1))
      
      (pRaw, trueFinal)
    }
    
    // 返回结果字典
    Map("pred" -> p, "true" -> trueValues)
  }
  
  // 计算损失
  def loss(feed_dict: Map[String, Tensor[ParamType]], out_dict: Map[String, Tensor[ParamType]]): (Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]) = {
    val pred = out_dict("pred").flatten()
    val trueValues = out_dict("true").flatten()
    
    // 创建掩码，过滤掉无效值
    val mask = trueValues > -1.0f
    
    // 计算损失
    val loss = loss_fn(pred(mask), trueValues(mask))
    
    // 返回损失值、有效样本数和正样本数
    (loss, mask.sum(), trueValues(mask).sum())
  }
  
  // 实现apply方法，调用forward
  override def apply(input: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = forward(input)

// 辅助函数：创建上三角掩码
def ut_mask[ParamType <: FloatNN: Default](seq_len: Long, device: Device): Tensor[ParamType] = {
  // 创建全1矩阵，然后转换为上三角矩阵，对角线为1
  val ones = torch.ones(Seq(seq_len, seq_len), device = device)
  torch.triu(ones, diagonal = 1)
}

// 辅助函数：位置编码
def pos_encode[ParamType <: FloatNN: Default](seq_len: Long, device: Device): Tensor[ParamType] = {
  // 创建从0到seq_len-1的序列作为位置编码
  torch.arange(seq_len, device = device).unsqueeze(0)
}

// 辅助函数：复制模块列表
def get_clones[ParamType <: FloatNN: Default](module: => Blocks[ParamType], N: Int): List[Blocks[ParamType]] = {
  // 创建N个模块的深拷贝列表
  (0 until N).map(_ => module).toList
}

// SAKT模型的工厂方法
object SAKT {
  def apply[ParamType <: FloatNN: Default](
    joint: Boolean,
    mask_future: Boolean,
    length: Int,
    trans: Boolean,
    num_skills: Int,
    seq_len: Int,
    embedding_size: Int,
    num_attn_heads: Int,
    dropout: Double,
    num_blocks: Int = 2
  ): SAKT[ParamType] = {
    new SAKT[ParamType](joint, mask_future, length, trans, num_skills, seq_len, embedding_size, num_attn_heads, dropout, num_blocks)
  }
}
