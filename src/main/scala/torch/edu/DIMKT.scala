package torch.edu

import torch.*
import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.*
import torch.nn as nn
import scala.collection.mutable.ListBuffer

class DIMKT[ParamType <: FloatNN: Default](
    mask_future: Boolean,
    length: Int,
    trans: Boolean,
    num_skills: Int,
    num_questions: Int,
    embedding_size: Int,
    dropout: Double,
    batch_size: Int,
    difficult_levels: Int = 100
) extends TensorModule[ParamType] with HasParams[ParamType] {

  // 模型参数
//  val mask_future = mask_future
//  val length = length
//  val trans = trans
//  val num_questions = num_questions
//  val num_skills = num_skills
//  val embedding_size = embedding_size
//  val batch_size = batch_size
//  val difficult_levels = difficult_levels
  val sigmoid = Sigmoid[ParamType]()
  val tanh = Tanh[ParamType]()
  val dropout = Dropout[ParamType](dropout)

  // 嵌入层定义
  val interaction_emb: Embedding[ParamType] = nn.Embedding[ParamType](num_skills * 2, embedding_size)

  // 知识参数
  val knowledge: nn.Parameter[ParamType] = {
    val initValue = torch.empty[ParamType](1, embedding_size)
    nn.init.xavier_uniform_(initValue)
    nn.Parameter[ParamType](initValue, requiresGrad = true)
  }

  // 各种嵌入层
  val q_emb: Embedding[ParamType] = nn.Embedding[ParamType](num_questions + 1, embedding_size, paddingIdx = 0)
  val c_emb: Embedding[ParamType] = nn.Embedding[ParamType](num_skills + 1, embedding_size, paddingIdx = 0)
  val sd_emb: Embedding[ParamType] = nn.Embedding[ParamType](difficult_levels + 2, embedding_size, paddingIdx = 0)
  val qd_emb: Embedding[ParamType] = nn.Embedding[ParamType](difficult_levels + 2, embedding_size, paddingIdx = 0)
  val a_emb: Embedding[ParamType] = nn.Embedding[ParamType](2, embedding_size)

  // 线性层
  val linear_1: Linear[ParamType] = nn.Linear[ParamType](4 * embedding_size, embedding_size)
  val linear_2: Linear[ParamType] = nn.Linear[ParamType](1 * embedding_size, embedding_size)
  val linear_3: Linear[ParamType] = nn.Linear[ParamType](1 * embedding_size, embedding_size)
  val linear_4: Linear[ParamType] = nn.Linear[ParamType](2 * embedding_size, embedding_size)
  val linear_5: Linear[ParamType] = nn.Linear[ParamType](2 * embedding_size, embedding_size)
  val linear_6: Linear[ParamType] = nn.Linear[ParamType](4 * embedding_size, embedding_size)

  // 损失函数
  val loss_fn: BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction = "mean")

  // 输出层
  val out: Sequential[ParamType] = if (trans || mask_future) {
    nn.Sequential[ParamType](
      nn.Linear[ParamType](embedding_size, embedding_size),
      nn.ReLU[ParamType](),
      nn.Dropout[ParamType](dropout),
      nn.Linear[ParamType](embedding_size, 256),
      nn.ReLU[ParamType](),
      nn.Dropout[ParamType](dropout),
      nn.Linear[ParamType](256, num_skills)
    )
  } else {
    nn.Sequential[ParamType](nn.Linear[ParamType](1, 1)) // 占位，不会被使用
  }

  // 前向传播
  def forward(feed_dict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = {
    val q = feed_dict("questions")
    val c = feed_dict("skills")
    val qd = feed_dict("question_difficulty")
    val sd = feed_dict("skill_difficulty")
    val r = feed_dict("responses")

    // 处理输入数据
    val q_input = q.slice(1, 0, -length)
    val c_input = c.slice(1, 0, -length)
    val sd_input = sd.slice(1, 0, -length)
    val qd_input = qd.slice(1, 0, -length)
    val masked_r = r * (r > Tensor(-1))
    val r_input = masked_r.slice(1, 0, -length)
    val rshft = masked_r.slice(1, length)
    val qshft = q.slice(1, length)
    val cshft = c.slice(1, length)
    val qdshft = qd.slice(1, length)
    val sdshft = sd.slice(1, length)

    // 更新batch_size
    if (batch_size != q_input.shape(0)) {
      self.batch_size = q_input.shape(0)
    }

    // 获取嵌入
    val q_emb_val = q_emb(q_input)
    val c_emb_val = c_emb(c_input)
    val sd_emb_val = sd_emb(sd_input)
    val qd_emb_val = qd_emb(qd_input)
    val a_emb_val = a_emb(r_input)
    val device = q_emb_val.device

    // 目标嵌入
    val target_q = q_emb(qshft)
    val target_c = c_emb(cshft)
    val target_sd = sd_emb(sdshft)
    val target_qd = qd_emb(qdshft)

    // 处理输入数据
    val input_data = torch.cat(List(q_emb_val, c_emb_val, sd_emb_val, qd_emb_val), dim = -1)
    val transformed_input = linear_1(input_data)

    // 处理目标数据
    val target_data = torch.cat(List(target_q, target_c, target_sd, target_qd), dim = -1)
    val transformed_target = linear_1(target_data)

    // 为序列数据添加padding并分割
    val shape_sd = sd_emb_val.shape
    val padd_sd = torch.zeros(shape_sd(0), 1, shape_sd(2), device = device)
    val padded_sd = torch.cat(List(padd_sd, sd_emb_val), dim = 1)
    val slice_sd_embedding = padded_sd.split(1, dim = 1)

    val shape_a = a_emb_val.shape
    val padd_a = torch.zeros(shape_a(0), 1, shape_a(2), device = device)
    val padded_a = torch.cat(List(padd_a, a_emb_val), dim = 1)
    val slice_a_embedding = padded_a.split(1, dim = 1)

    val shape_input = transformed_input.shape
    val padd_input = torch.zeros(shape_input(0), 1, shape_input(2), device = device)
    val padded_input = torch.cat(List(padd_input, transformed_input), dim = 1)
    val slice_input_data = padded_input.split(1, dim = 1)

    val shape_qd = qd_emb_val.shape
    val padd_qd = torch.zeros(shape_qd(0), 1, shape_qd(2), device = device)
    val padded_qd = torch.cat(List(padd_qd, qd_emb_val), dim = 1)
    val slice_qd_embedding = padded_qd.split(1, dim = 1)

    // 初始化知识向量
    var k = knowledge.value.repeat(batch_size, 1).to(device)

    // 处理序列
    val h = ListBuffer[Tensor[ParamType]]()
    val seqlen = q_input.shape(1)
    for (i <- 1 to seqlen) {
      val sd_1 = slice_sd_embedding(i).squeeze(1)
      val a_1 = slice_a_embedding(i).squeeze(1)
      val qd_1 = slice_qd_embedding(i).squeeze(1)
      val input_data_1 = slice_input_data(i).squeeze(1)

      val qq = k - input_data_1

      val gates_SDF = sigmoid(linear_2(qq))
      val SDFt = tanh(linear_3(qq))
      val SDFt_dropped = dropout(SDFt)

      val SDFt_final = gates_SDF * SDFt_dropped

      val x = torch.cat(List(SDFt_final, a_1), dim = -1)
      val gates_PKA = sigmoid(linear_4(x))

      val PKAt = tanh(linear_5(x))

      val PKAt_final = gates_PKA * PKAt

      val ins = torch.cat(List(k, a_1, sd_1, qd_1), dim = -1)
      val gates_KSU = sigmoid(linear_6(ins))

      k = gates_KSU * k + (Tensor(1) - gates_KSU) * PKAt_final

      val h_i = k.unsqueeze(1)
      h.append(h_i)
    }

    val output = torch.cat(h.toList, dim = 1)

    // 根据不同模式计算输出
    val (logits, true_val) = if (trans) {
      val logits_val = sigmoid(out(transformed_target * output))
      val one_hot_cshft = F.one_hot(cshft, num_skills)
      val logits_sum = (logits_val * one_hot_cshft).sum(dim = -1)
      val true_val = r.slice(1, length)
      (logits_sum, true_val)
    } else if (mask_future) {
      val logits_val = sigmoid(out(transformed_target * output))
      val one_hot_cshft = F.one_hot(cshft, num_skills)
      val logits_sum = (logits_val * one_hot_cshft).sum(dim = -1).slice(1, -length)
      val true_val = r.slice(1, -length)
      (logits_sum, true_val)
    } else {
      val logits_val = sigmoid(torch.sum(transformed_target * output, dim = -1))
      val true_val = r.slice(1, length)
      (logits_val, true_val)
    }

    // 返回结果
    Map(
      "pred" -> logits,
      "true" -> true_val
    )
  }

  // 损失计算
  def loss(feed_dict: Map[String, Tensor[ParamType]], out_dict: Map[String, Tensor[ParamType]]): (Tensor[ParamType], Tensor[Long], Tensor[ParamType]) = {
    val pred = out_dict("pred").flatten()
    val true_ = out_dict("true").flatten()
    val mask = true_ > Tensor(-1)
    val loss = loss_fn(pred.masked_select(mask), true_.masked_select(mask))
    (loss, Tensor(mask.sum().item().long()), true_.masked_select(mask).sum())
  }

  // 实现apply方法
  def apply(input: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = forward(input)
}

// 伴生对象，提供工厂方法
object DIMKT {
  def apply[ParamType <: FloatNN: Default](
    mask_future: Boolean,
    length: Int,
    trans: Boolean,
    num_skills: Int,
    num_questions: Int,
    embedding_size: Int,
    dropout: Double,
    batch_size: Int,
    difficult_levels: Int = 100
  ): DIMKT[ParamType] = {
    new DIMKT(
      mask_future, length, trans, num_skills, num_questions, embedding_size,
      dropout, batch_size, difficult_levels
    )
  }
}
