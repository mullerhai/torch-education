package torch.edu

import torch.*
import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}

import scala.collection.mutable.ListBuffer

class DeepIRT[ParamType <: FloatNN: Default](
    mask_response: Boolean,
    pred_last: Boolean,
    mask_future: Boolean,
    length: Int,
    trans: Boolean,
    num_skills: Int,
    dim_s: Int,
    size_m: Int,
    dropout: Double = 0.2
) extends HasParams[ParamType] with TensorModule[ParamType]:
  
  // 模型参数
  val k_emb_layer: Embedding[ParamType] = Embedding(num_skills, dim_s)
  val Mk: Tensor[ParamType] = Tensor(size_m, dim_s).requiresGrad_(true)
  val Mv0: Tensor[ParamType] = Tensor(size_m, dim_s).requiresGrad_(true)
  
  // 初始化参数
  kaiming_normal_(Mk)
  kaiming_normal_(Mv0)
  
  val v_emb_layer: Embedding[ParamType] = Embedding(num_skills * 2, dim_s)
  val f_layer: Linear[ParamType] = Linear(dim_s * 2, dim_s)
  val dropout_layer: Dropout[ParamType] = Dropout(dropout)
  val p_layer: Linear[ParamType] = Linear(dim_s, 1)
  
  // 根据trans参数配置不同的层
  val diff_layer: Sequential[ParamType] = 
    if (trans) Sequential(Linear(dim_s, num_skills), Tanh())
    else Sequential(Linear(dim_s, 1), Tanh())
  
  val ability_layer: Sequential[ParamType] = 
    if (trans) Sequential(Linear(dim_s, num_skills), Tanh())
    else Sequential(Linear(dim_s, 1), Tanh())
  
  val e_layer: Linear[ParamType] = Linear(dim_s, dim_s)
  val a_layer: Linear[ParamType] = Linear(dim_s, dim_s)
  val loss_fn: BCELoss = BCELoss(reduction = "mean")
  
  // 收集所有参数
  override val params: Seq[Tensor[ParamType]] = Seq(
    k_emb_layer, v_emb_layer, f_layer, dropout_layer, p_layer,
    diff_layer, ability_layer, e_layer, a_layer, Mk, Mv0
  ).flatMap(_.params)
  
  // apply方法调用forward
  def apply(feedDict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = forward(feedDict)
  
  // 前向传播
  def forward(feedDict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = {
    var q = feedDict("skills")
    val r = feedDict("responses")
    var masked_r = r * (r > Tensor(-1f))
    
    var cshft: Tensor[ParamType] = null
    
    // 根据不同模式处理输入
    if (trans) {
      cshft = q.slice(1, length, None)
      q = q.slice(1, 0, -length)
      masked_r = masked_r.slice(1, 0, -length)
    } else if (mask_future) {
      val attention_mask = feedDict("attention_mask").clone()
      attention_mask.slice(1, -length, None).fill_(0f)
      q = q * attention_mask
      masked_r = r * attention_mask
    } else if (mask_response) {
      val attention_mask = feedDict("attention_mask").clone()
      attention_mask.slice(1, -length, None).fill_(0f)
      masked_r = r * attention_mask
    }
    
    val batch_size = q.shape(0)
    val x = q + Tensor(num_skills.toFloat) * masked_r
    
    // 获取嵌入
    val k = k_emb_layer(q) // 问题嵌入
    val v = v_emb_layer(x) // 问题-答案嵌入
    
    // 初始化记忆矩阵
    var Mvt = Mv0.unsqueeze(0).repeat(batch_size, 1, 1)
    val Mv = ListBuffer[Tensor[ParamType]]()
    Mv.append(Mvt)
    
    // 计算权重
    val w = (k.matmul(Mk.t())).softmax(dim = -1)
    
    // 写入过程
    val e = e_layer(v).sigmoid()
    val a = a_layer(v).tanh()
    
    // 遍历序列长度
    for (i <- 0 until e.shape(1)) {
      val et = e.slice(1, i, i + 1).squeeze(1)
      val at = a.slice(1, i, i + 1).squeeze(1)
      val wt = w.slice(1, i, i + 1).squeeze(1)
      
      // 更新记忆矩阵
      Mvt = Mvt * (Tensor(1f) - (wt.unsqueeze(-1) * et.unsqueeze(1))) + 
            (wt.unsqueeze(-1) * at.unsqueeze(1))
      Mv.append(Mvt)
    }
    
    // 堆叠记忆矩阵
    val MvStacked = torch.stack(Mv.toSeq, dim = 1)
    
    // 读取过程
    val w_unsqueezed = w.unsqueeze(-1)
    val MvSliced = MvStacked.slice(1, 0, -1)
    val weightedSum = (w_unsqueezed * MvSliced).sum(dim = -2)
    
    // 拼接特征
    val combined = torch.cat(Seq(weightedSum, k), dim = -1)
    val f = f_layer(combined).tanh()
    
    // 计算学生能力和问题难度
    val stu_ability = ability_layer(dropout_layer(f)) // 公式12
    val que_diff = diff_layer(dropout_layer(k))       // 公式13
    
    // 计算预测概率
    val p = (Tensor(3.0f) * stu_ability - que_diff).sigmoid() // 公式14
    
    var pred: Tensor[ParamType] = null
    var trueVals: Tensor[ParamType] = null
    
    // 根据不同模式处理输出
    if (trans) {
      val cshftLong = cshft
      val oneHot = F.oneHot(cshftLong, num_skills)
      pred = (p * oneHot).sum(dim = -1)
      trueVals = r.slice(1, length, None)
    } else if (mask_future || pred_last || mask_response) {
      pred = p.squeeze(-1).slice(1, -length, None)
      trueVals = r.slice(1, -length, None)
    } else {
      pred = p.squeeze(-1).slice(1, length, None)
      trueVals = r.slice(1, length, None)
    }
    
    // 返回结果字典
    Map(
      "pred" -> pred,
      "true" -> trueVals
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
    val loss = loss_fn(maskedPred, maskedTrue)
    
    // 返回损失值、有效样本数和真实值总和
    (loss, maskedPred.size(0), maskedTrue.sum().item().asInstanceOf[Float])
  }
  
  // 辅助函数：Kaiming正态初始化
  private def kaiming_normal_(tensor: Tensor[ParamType]): Unit = {
    // 在Storch中，我们可以直接使用内置的初始化方法
    tensor.normal_(0f, math.sqrt(2.0 / tensor.shape.last).toFloat)
  }
