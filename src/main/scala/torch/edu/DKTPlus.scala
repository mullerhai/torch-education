package torch.edu

import torch.*
import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.*
import torch.nn as nn
import scala.collection.mutable.ListBuffer

class DKTPlus[ParamType <: FloatNN : Default](
    mask_future: Boolean,
    length: Int,
    num_skills: Int,
    lambda_r: Double = 0.01,
    lambda_w1: Double = 0.003,
    lambda_w2: Double = 3.0,
    embedding_size: Int = 64,
    dropout: Double = 0.1
) extends HasParams[ParamType] with TensorModule[ParamType] {

  // 初始化参数
  val emb_size: Int = embedding_size
  val hidden_size: Int = embedding_size
  val interaction_emb: Embedding[ParamType] = nn.Embedding(num_skills * 2, emb_size)
  val lstm_layer: LSTM[ParamType] = nn.LSTM(emb_size, hidden_size, batch_first = true)
  val dropout_layer: Dropout[ParamType] = nn.Dropout(dropout)
  val out_layer: Linear[ParamType] = nn.Linear(hidden_size, num_skills)
  val loss_fn: BCELoss = nn.BCELoss(reduction = "mean")

  // 收集所有参数
  override val params: ListBuffer[Tensor[ParamType]] = ListBuffer(
    interaction_emb.weight,
    lstm_layer.weight_ih_l0,
    lstm_layer.weight_hh_l0,
    lstm_layer.bias_ih_l0,
    lstm_layer.bias_hh_l0,
    out_layer.weight,
    out_layer.bias
  )

  def forward(feed_dict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = {
    val q = feed_dict("skills")
    val r = feed_dict("responses")
    val attention_mask = feed_dict("attention_mask").narrow(1, length, q.size(1) - length)
    
    // 处理响应掩码
    val masked_r = r * (r > Tensor(-1.0))
    val q_input = q.narrow(1, 0, q.size(1) - length)
    val r_input = masked_r.narrow(1, 0, masked_r.size(1) - length)
    
    // 计算输入特征
    val x = q_input + Tensor(num_skills) * r_input
    val xemb = interaction_emb.forward(x)
    
    // LSTM处理
    val (h, _) = lstm_layer.forward(xemb)
    val h_dropout = dropout_layer.forward(h)
    val y = out_layer.forward(h_dropout)
    val y_sigmoid = torch.sigmoid(y)
    
    // 计算预测和真实值
    val (pred, true_r) = if (mask_future) {
      val pred_val = (y_sigmoid * F.one_hot(q.narrow(1, length, q.size(1) - length), num_skills)).sum(-1)
      val pred_narrow = pred_val.narrow(1, pred_val.size(1) - length, length)
      val r_shft = r.narrow(1, length, r.size(1) - length)
      val true_val = r_shft.narrow(1, r_shft.size(1) - length, length)
      (pred_narrow, true_val)
    } else {
      val pred_val = (y_sigmoid * F.one_hot(q.narrow(1, length, q.size(1) - length), num_skills)).sum(-1)
      val true_val = r.narrow(1, length, r.size(1) - length)
      (pred_val, true_val)
    }
    
    // 构造输出字典
    Map(
      "pred" -> pred,
      "true" -> true_r,
      "y" -> y_sigmoid,
      "y_curr" -> (y_sigmoid * F.one_hot(q_input, num_skills)).sum(-1),
      "y_next" -> (y_sigmoid * F.one_hot(q.narrow(1, length, q.size(1) - length), num_skills)).sum(-1),
      "r_curr" -> r_input,
      "r_next" -> r.narrow(1, length, r.size(1) - length),
      "attention_mask" -> attention_mask
    )
  }

  def loss(feed_dict: Map[String, Tensor[ParamType]], out_dict: Map[String, Tensor[ParamType]]): (Tensor[ParamType], Int, Double) = {
    val sm = out_dict("attention_mask")
    val y = out_dict("y")
    val y_curr = out_dict("y_curr")
    val y_next = out_dict("y_next")
    val r_curr = out_dict("r_curr")
    val r_next = out_dict("r_next")
    
    // 应用掩码并计算损失
    val y_curr_masked = torch.masked_select(y_curr, sm.bools())
    val y_next_masked = torch.masked_select(y_next, sm.bools())
    val r_curr_masked = torch.masked_select(r_curr, sm.bools())
    val r_next_masked = torch.masked_select(r_next, sm.bools())
    
    // 主损失
    val loss_main = loss_fn.forward(y_next_masked, r_next_masked)
    
    // 正则化损失项
    val loss_r = loss_fn.forward(y_curr_masked, r_curr_masked)
    
    // L1正则化
    val y_shifted_1 = y.narrow(1, length, y.size(1) - length)
    val y_shifted_2 = y.narrow(1, 0, y.size(1) - length)
    val loss_w1 = torch.masked_select(
      torch.norm(y_shifted_1 - y_shifted_2, p = 1, dim = -1), 
      sm.narrow(1, length, sm.size(1) - length)
    ).mean() / num_skills
    
    // L2正则化的平方
    val loss_w2 = torch.masked_select(
      torch.norm(y_shifted_1 - y_shifted_2, p = 2, dim = -1).pow(2), 
      sm.narrow(1, length, sm.size(1) - length)
    ).mean() / num_skills
    
    // 总损失
    val total_loss = loss_main + Tensor(lambda_r) * loss_r +
                    Tensor(lambda_w1) * loss_w1 +
                    Tensor(lambda_w2) * loss_w2
    
    (total_loss, y_next_masked.size(0), r_next_masked.sum().item())
  }

  // 实现apply方法，默认调用forward
  override def apply(input: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = {
    forward(input)
  }
}

// 伴生对象，用于创建模型实例
object DKTPlus {
  def apply[ParamType <: FloatNN : Default](
      mask_future: Boolean,
      length: Int,
      num_skills: Int,
      lambda_r: Double = 0.01,
      lambda_w1: Double = 0.003,
      lambda_w2: Double = 3.0,
      embedding_size: Int = 64,
      dropout: Double = 0.1
  ): DKTPlus[ParamType] = {
    new DKTPlus(mask_future, length, num_skills, lambda_r, lambda_w1, lambda_w2, embedding_size, dropout)
  }
}
