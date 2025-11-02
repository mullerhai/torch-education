package torch.edu

import torch.*
import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn as nn
import torch.nn.*
import scala.collection.mutable.ListBuffer

class DKT[ParamType <: FloatNN: Default](
    joint: Boolean,
    maskFuture: Boolean,
    length: Int,
    numSkills: Int,
    embeddingSize: Int = 64,
    dropout: Double = 0.1
) extends HasParams[ParamType] with TensorModule[ParamType]:
  
  // 模型参数
  val embSize: Int = embeddingSize
  val hiddenSize: Int = embeddingSize
  
  // 模型层
  val interactionEmb: Embedding[ParamType] = Embedding(numSkills * 2, embSize)
  val lstmLayer: LSTM[ParamType] = LSTM(embSize, hiddenSize, batchFirst = true)
  val dropoutLayer: Dropout[ParamType] = Dropout(dropout)
  val outLayer: Linear[ParamType] = Linear(hiddenSize, numSkills)
  val lossFn: BCELoss = BCELoss(reduction = "mean")
  
  // 收集所有参数
  override val params: Seq[Tensor[ParamType]] = Seq(
    interactionEmb, 
    lstmLayer, 
    dropoutLayer, 
    outLayer
  ).flatMap(_.params)
  
  // apply方法调用forward
  def apply(feedDict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = forward(feedDict)
  
  // 前向传播
  def forward(feedDict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = {
    val q = feedDict("skills")
    val r = feedDict("responses")
    
    // 处理masked_r
    val maskedR = r * (r > Tensor(-1f))
    
    // 准备输入数据
    val qInput = q.slice(1, 0, -length)
    val rInput = maskedR.slice(1, 0, -length)
    val qShft = q.slice(1, length, None)
    val rShft = r.slice(1, length, None)
    
    // 计算交互嵌入
    val x = qInput + Tensor(numSkills.toFloat) * rInput
    val xemb = interactionEmb(x)
    
    // LSTM处理
    val (h, _) = lstmLayer(xemb)
    val hDropout = dropoutLayer(h)
    var y = outLayer(hDropout)
    
    // 根据joint参数处理
    var mid = 0
    if (joint) {
      val seqLen = qInput.size(1)
      mid = seqLen / 2
      // 扩展中间结果
      val midSlice = y.slice(1, mid, mid + 1)
      val expanded = midSlice.expand(-1, seqLen - mid, -1)
      y = y.slice(1, 0, mid).cat(expanded, dim = 1)
    }
    
    // 应用sigmoid激活
    y = y.sigmoid()
    
    // 与one-hot编码相乘并求和
    val qShftLong = qShft
    val oneHot = F.oneHot(qShftLong, numSkills)
    val ySum = (y * oneHot).sum(dim = -1)
    
    // 根据mask_future和joint参数选择输出部分
    var yOut = ySum
    var rShftOut = rShft
    
    if (maskFuture) {
      yOut = yOut.slice(1, -length, None)
      rShftOut = rShftOut.slice(1, -length, None)
    } else if (joint) {
      yOut = yOut.slice(1, mid, None)
      rShftOut = rShftOut.slice(1, mid, None)
    }
    
    // 返回结果字典
    Map(
      "pred" -> yOut,
      "true" -> rShftOut
    )
  }
  
  // 损失计算
  def loss(feedDict: Map[String, Tensor[ParamType]], outDict: Map[String, Tensor[ParamType]]): (Tensor[ParamType], Int, Float) = {
    val pred = outDict("pred").flatten()
    val trueVals = outDict("true").flatten()
    
    // 计算mask
    val mask = trueVals > Tensor(-1f)
    
    // 应用mask并计算损失
    val maskedPred = pred.masked_select(mask)
    val maskedTrue = trueVals.masked_select(mask)
    val loss = lossFn(maskedPred, maskedTrue)
    
    // 返回损失值、有效样本数和真实值总和
    (loss, maskedPred.size(0), maskedTrue.sum().item().asInstanceOf[Float])
  }
