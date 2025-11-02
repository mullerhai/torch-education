package torch.edu

import torch.*
import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn as nn
import torch.nn.*
import scala.collection.mutable.ListBuffer

class DKVMN[ParamType <: FloatNN: Default] private (
    val joint: Boolean,
    val mask_response: Boolean,
    val pred_last: Boolean,
    val mask_future: Boolean,
    val length: Int,
    val trans: Boolean,
    val num_skills: Int,
    val dim_s: Int,
    val size_m: Int,
    val dropout: Double = 0.2
) extends TensorModule[ParamType] with HasParams[ParamType] {

  // 层定义
  val k_emb_layer = nn.Embedding[ParamType](num_skills, dim_s)
  val Mk = nn.Parameter[ParamType](torch.Tensor[ParamType](size_m, dim_s))
  val Mv0 = nn.Parameter[ParamType](torch.Tensor[ParamType](size_m, dim_s))

  // 初始化参数
  nn.init.kaiming_normal_(Mk)
  nn.init.kaiming_normal_(Mv0)

  val v_emb_layer = nn.Embedding[ParamType](num_skills * 2, dim_s)

  val f_layer = nn.Linear[ParamType](dim_s * 2, dim_s)
  val dropout_layer = nn.Dropout[ParamType](dropout)
  val p_layer = if (trans) {
    nn.Linear[ParamType](dim_s, num_skills)
  } else {
    nn.Linear[ParamType](dim_s, 1)
  }

  val e_layer = nn.Linear[ParamType](dim_s, dim_s)
  val a_layer = nn.Linear[ParamType](dim_s, dim_s)
  val loss_fn = nn.BCELoss(reduction = "mean")

  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)

  def forward(feed_dict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = {
    val q = feed_dict("skills")
    val r = feed_dict("responses")
    val masked_r = r * (r > Tensor(-1.0))

    val (q_input, r_input, cshft) = if (trans) {
      val cshft = q.narrow(1, length, q.shape(1) - length)
      val q_input = q.narrow(1, 0, q.shape(1) - length)
      val r_input = masked_r.narrow(1, 0, masked_r.shape(1) - length)
      (q_input, r_input, Some(cshft))
    } else if (mask_future) {
      val attention_mask = feed_dict("attention_mask")
      // 创建一个新的mask张量，避免直接修改输入
      val new_mask = attention_mask.clone()
      // 设置最后length个位置为0
      val last_part = new_mask.narrow(1, new_mask.shape(1) - length, length)
      last_part.fill_(0.0f)
      val q_input = q * new_mask
      val r_input = r * new_mask
      (q_input, r_input, None)
    } else if (mask_response) {
      val attention_mask = feed_dict("attention_mask")
      // 创建一个新的mask张量，避免直接修改输入
      val new_mask = attention_mask.clone()
      // 设置最后length个位置为0
      val last_part = new_mask.narrow(1, new_mask.shape(1) - length, length)
      last_part.fill_(0.0f)
      val r_input = r * new_mask
      (q, r_input, None)
    } else {
      (q, masked_r, None)
    }

    val batch_size = q.shape(0)
    val x = q_input + Tensor(num_skills) * r_input
    val k = k_emb_layer(q_input)
    val v = v_emb_layer(x)

    var Mvt = Mv0.unsqueeze(0).repeat(Seq(batch_size, 1, 1))

    val Mv = ListBuffer[Tensor[ParamType]]()
    Mv += Mvt

    val w = F.softmax(torch.matmul(k, Mk.t), dim = -1)

    // Write Process
    val e = F.sigmoid(e_layer(v))
    val a = F.tanh(a_layer(v))

    // 交换维度以便按时间步处理
    val e_permuted = e.permute(Seq(1, 0, 2))
    val a_permuted = a.permute(Seq(1, 0, 2))
    val w_permuted = w.permute(Seq(1, 0, 2))

    // 获取时间步数量
    val time_steps = e_permuted.shape(0)

    for (t <- 0 until time_steps) {
      val et = e_permuted.select(0, t)
      val at = a_permuted.select(0, t)
      val wt = w_permuted.select(0, t)
      
      val wt_unsqueeze = wt.unsqueeze(-1)
      val et_unsqueeze = et.unsqueeze(1)
      val at_unsqueeze = at.unsqueeze(1)
      
      Mvt = Mvt * (Tensor(1.0f) - (wt_unsqueeze * et_unsqueeze)) + (wt_unsqueeze * at_unsqueeze)
      Mv += Mvt
    }

    val Mv_stack = torch.stack(Mv.toSeq, dim = 1)

    // Read Process
    val w_unsqueeze = w.unsqueeze(-1)
    val Mv_sliced = Mv_stack.narrow(1, 0, Mv_stack.shape(1) - 1)
    val weighted_sum = (w_unsqueeze * Mv_sliced).sum(dim = -2)
    
    val f_input = torch.cat(Seq(weighted_sum, k), dim = -1)
    val f = F.tanh(f_layer(f_input))
    val p = p_layer(dropout_layer(f))

    // 处理joint情况
    if (joint) {
      val seq_len = p.shape(1)
      val mid = seq_len / 2
      val mid_slice = p.narrow(1, mid, 1)
      val expanded_mid = mid_slice.expand(Seq(-1, seq_len - mid, -1))
      // 创建一个副本并修改
      val p_modified = p.clone()
      p_modified.narrow(1, mid, seq_len - mid).copy_(expanded_mid)
    }

    val p_sigmoid = F.sigmoid(p)

    // 根据不同的模式处理输出
    val (p_final, true_final) = if (trans) {
      val cshft_long = cshft.get.toType(torch.int64)
      val one_hot = F.one_hot(cshft_long, num_skills)
      val p_sum = (p_sigmoid * one_hot).sum(dim = -1)
      
      if (joint) {
        val seq_len = p_sum.shape(1)
        val mid = seq_len / 2
        val p_sliced = p_sum.narrow(1, mid, seq_len - mid)
        val r_shft = r.narrow(1, length, r.shape(1) - length)
        val true_sliced = r_shft.narrow(1, mid, r_shft.shape(1) - mid)
        (p_sliced, true_sliced)
      } else {
        val r_shft = r.narrow(1, length, r.shape(1) - length)
        (p_sum, r_shft)
      }
    } else if (mask_future || pred_last || mask_response) {
      val p_squeezed = p_sigmoid.squeeze(-1)
      val p_sliced = p_squeezed.narrow(1, p_squeezed.shape(1) - length, length)
      val true_sliced = r.narrow(1, r.shape(1) - length, length)
      (p_sliced, true_sliced)
    } else {
      val p_squeezed = p_sigmoid.squeeze(-1)
      val p_sliced = p_squeezed.narrow(1, length, p_squeezed.shape(1) - length)
      val true_sliced = r.narrow(1, length, r.shape(1) - length)
      (p_sliced, true_sliced)
    }

    Map(
      "pred" -> p_final,
      "true" -> true_final
    )
  }

  def loss(feed_dict: Map[String, Tensor[ParamType]], out_dict: Map[String, Tensor[ParamType]]): (Tensor[ParamType], Long, Tensor[ParamType]) = {
    val pred = out_dict("pred").flatten()
    val true_tensor = out_dict("true").flatten()
    val mask = true_tensor > Tensor(-1.0)
    
    // 应用掩码
    val masked_pred = pred.masked_select(mask)
    val masked_true = true_tensor.masked_select(mask)
    
    val loss = loss_fn(masked_pred, masked_true)
    val count = masked_pred.numel()
    val sum_true = masked_true.sum()
    
    (loss, count, sum_true)
  }
}

object DKVMN {
  def apply[ParamType <: FloatNN: Default](
      joint: Boolean,
      mask_response: Boolean,
      pred_last: Boolean,
      mask_future: Boolean,
      length: Int,
      trans: Boolean,
      num_skills: Int,
      dim_s: Int,
      size_m: Int,
      dropout: Double = 0.2
  ): DKVMN[ParamType] = {
    new DKVMN[ParamType](joint, mask_response, pred_last, mask_future, length, trans, num_skills, dim_s, size_m, dropout)
  }
}
