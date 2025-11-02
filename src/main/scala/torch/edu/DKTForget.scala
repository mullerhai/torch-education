package torch.edu

import torch.*
import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.*
import torch.nn as nn
import scala.collection.mutable.ListBuffer

// CIntegration类的实现
class CIntegration[ParamType <: FloatNN: Default](
    device: Device,
    num_rgap: Int,
    num_sgap: Int,
    num_pcount: Int,
    emb_dim: Int
) extends TensorModule[ParamType] with HasParams[ParamType]:
  // 初始化单位矩阵
  val rgap_eye: Tensor[ParamType] = torch.eye(num_rgap).to(device)
  val sgap_eye: Tensor[ParamType] = torch.eye(num_sgap).to(device)
  val pcount_eye: Tensor[ParamType] = torch.eye(num_pcount).to(device)

  val ntotal: Int = num_rgap + num_sgap + num_pcount
  val cemb: Linear[ParamType] = Linear(ntotal, emb_dim, bias = false)

  // 打印调试信息
  println(s"num_sgap: $num_sgap, num_rgap: $num_rgap, num_pcount: $num_pcount, ntotal: $ntotal")

  def forward(vt: Tensor[ParamType], rgap: Tensor[ParamType], sgap: Tensor[ParamType], pcount: Tensor[ParamType]): Tensor[ParamType] = {
    // 获取one-hot编码
    val rgap_embed = rgap_eye.index_select(0, rgap.long())
    val sgap_embed = sgap_eye.index_select(0, sgap.long())
    val pcount_embed = pcount_eye.index_select(0, pcount.long())

    // 连接特征
    val ct = torch.cat(Seq(rgap_embed, sgap_embed, pcount_embed), dim = -1)
    
    // 应用线性变换
    val cct = cemb(ct)
    
    // 元素级乘法并连接
    val theta = torch.mul(vt, cct)
    torch.cat(Seq(theta, ct), dim = -1)
  }

  // apply方法调用forward
  def apply(vt: Tensor[ParamType], rgap: Tensor[ParamType], sgap: Tensor[ParamType], pcount: Tensor[ParamType]): Tensor[ParamType] = {
    forward(vt, rgap, sgap, pcount)
  }

// DKTForget主模型类的实现
class DKTForget[ParamType <: FloatNN: Default](
    mask_future: Boolean,
    length: Int,
    device: Device,
    num_skills: Int,
    num_rgap: Int,
    num_sgap: Int,
    num_pcount: Int,
    embedding_size: Int,
    dropout: Double = 0.1
) extends TensorModule[ParamType] with HasParams[ParamType]:
  val hidden_size: Int = embedding_size
  val ntotal: Int = num_rgap + num_sgap + num_pcount

  // 交互嵌入层
  val interaction_emb: Embedding[ParamType] = Embedding(num_skills * 2, embedding_size)

  // CIntegration层
  val c_integration: CIntegration[ParamType] = CIntegration(
    device, num_rgap, num_sgap, num_pcount, embedding_size
  )

  // LSTM层
  val lstm_layer: LSTM[ParamType] = LSTM(embedding_size + ntotal, hidden_size, batchFirst = true)

  // Dropout层
  val dropout_layer: Dropout[ParamType] = Dropout(dropout)

  // 输出层
  val out_layer: Linear[ParamType] = Linear(hidden_size + ntotal, num_skills)

  // 损失函数
  val loss_fn: BCELoss = BCELoss(reduction = "mean")

  def forward(feed_dict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = {
    // 获取输入数据
    val q = feed_dict("skills")
    val r = feed_dict("responses")
    val attention_mask = feed_dict("attention_mask")
    
    // 处理响应数据
    val masked_r = r * (r > Tensor(-1.0f))
    
    // 准备输入和移位后的输入
    val q_input = q.slice(1, 0, -length)
    val r_input = masked_r.slice(1, 0, -length)
    val q_shft = q.slice(1, length, -1)
    val r_shft = r.slice(1, length, -1)
    
    // 创建交互特征
    val x = q_input + Tensor(num_skills.toFloat) * r_input
    
    // 获取时间特征
    val rgaps = feed_dict("rgaps")
    val sgaps = feed_dict("sgaps")
    val pcounts = feed_dict("pcounts")
    
    val rgaps_input = rgaps.slice(1, 0, -length)
    val sgaps_input = sgaps.slice(1, 0, -length)
    val pcounts_input = pcounts.slice(1, 0, -length)
    
    val rgaps_shft = rgaps.slice(1, length, -1)
    val sgaps_shft = sgaps.slice(1, length, -1)
    val pcounts_shft = pcounts.slice(1, length, -1)
    
    // 获取交互嵌入
    val xemb = interaction_emb(x)
    
    // 应用CIntegration层
    val theta_in = c_integration(xemb, rgaps_input, sgaps_input, pcounts_input)
    
    // LSTM处理
    val (h, _) = lstm_layer(theta_in)
    
    // 输出CIntegration层
    val theta_out = c_integration(h, rgaps_shft, sgaps_shft, pcounts_shft)
    
    // 应用dropout和输出层
    val dropout_out = dropout_layer(theta_out)
    var y = out_layer(dropout_out)
    
    // 应用sigmoid激活函数
    val sigmoid = Sigmoid[ParamType]()
    y = sigmoid(y)
    
    // 使用one-hot编码选择对应的技能输出
    val one_hot_skills = F.one_hot(q_shft.long(), num_skills)
    y = (y * one_hot_skills).sum(-1)
    
    // 如果需要，应用未来掩码
    var y_final = y
    var r_shft_final = r_shft
    if (mask_future) {
      y_final = y_final.slice(1, -length, -1)
      r_shft_final = r_shft_final.slice(1, -length, -1)
    }
    
    // 返回结果
    Map(
      "pred" -> y_final,
      "true" -> r_shft_final
    )
  }

  // apply方法调用forward
  def apply(feed_dict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = {
    forward(feed_dict)
  }

  // 损失计算方法
  def loss(feed_dict: Map[String, Tensor[ParamType]], out_dict: Map[String, Tensor[ParamType]]): (Tensor[ParamType], Int, Double) = {
    val pred = out_dict("pred").flatten()
    val true_values = out_dict("true").flatten()
    
    // 创建掩码
    val mask = true_values > Tensor(-1.0f)
    
    // 计算损失
    val loss_value = loss_fn(pred.masked_select(mask), true_values.masked_select(mask))
    
    // 返回损失值、掩码数量和真实值总和
    (
      loss_value,
      pred.masked_select(mask).numel,
      true_values.masked_select(mask).sum().double()
    )
  }

// 伴生对象，提供工厂方法
object DKTForget {
  def apply[ParamType <: FloatNN: Default](
    mask_future: Boolean,
    length: Int,
    device: Device,
    num_skills: Int,
    num_rgap: Int,
    num_sgap: Int,
    num_pcount: Int,
    embedding_size: Int,
    dropout: Double = 0.1
  ): DKTForget[ParamType] = {
    new DKTForget(
      mask_future, length, device, num_skills, num_rgap, num_sgap, num_pcount,
      embedding_size, dropout
    )
  }
}
