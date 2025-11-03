package torch.edu

import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.{modules, functional as F}

import java.util.concurrent.atomic.AtomicInteger
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

// 定义维度枚举类
enum Dim: 
  case batch, seq, feature

// 主模型类
class folibiKT[ParamType <: FloatNN: Default] 
    (mask_response: Boolean, 
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
     d_ff: Int = 256, 
     kq_same: Boolean = true, 
     final_fc_dim: Int = 512, 
     num_attn_heads: Int = 8, 
     separate_qr: Boolean = false, 
     l2: Double = 1e-5, 
     emb_type: String = "qid_alibi", 
     num_buckets: Int = 32, 
     max_distance: Int = 100)
    extends TensorModule[ParamType] 
    with HasParams[ParamType] {

  // 模型参数
  val model_name = "folibikt"
  val model_type = model_name
  val embed_l = embedding_size

  // 嵌入层
  var difficult_param: Option[nn.Embedding[ParamType]] = None
  var q_embed_diff: Option[nn.Embedding[ParamType]] = None
  var qa_embed_diff: Option[nn.Embedding[ParamType]] = None

  var q_embed: Option[nn.Embedding[ParamType]] = None
  var qa_embed: Option[nn.Embedding[ParamType]] = None

  // 根据参数初始化嵌入层
  if (num_questions > 0) {
    difficult_param = Some(nn.Embedding[ParamType](num_questions + 1, 1))
    q_embed_diff = Some(nn.Embedding[ParamType](num_skills + 1, embed_l))
    qa_embed_diff = Some(nn.Embedding[ParamType](2 * num_skills + 1, embed_l))
  }

  if (emb_type.startsWith("qid")) {
    q_embed = Some(nn.Embedding[ParamType](num_skills, embed_l))
    if (separate_qr) {
      qa_embed = Some(nn.Embedding[ParamType](2 * num_skills + 1, embed_l))
    } else {
      qa_embed = Some(nn.Embedding[ParamType](2, embed_l))
    }
  }

  // 架构对象
  val model = new Architecture4[ParamType](
    num_skills = num_skills,
    num_blocks = num_blocks,
    n_heads = num_attn_heads,
    dropout = dropout,
    embedding_size = embedding_size,
    d_feature = embedding_size / num_attn_heads,
    d_ff = d_ff,
    kq_same = kq_same,
    model_type = model_type,
    seq_len = seq_len,
    emb_type = emb_type,
    num_buckets = num_buckets,
    max_distance = max_distance
  )

  // 输出层
  val out = if (trans) {
    nn.Sequential[ParamType](
      nn.Linear[ParamType](embedding_size + embed_l, final_fc_dim),
      nn.ReLU[ParamType](),
      nn.Dropout[ParamType](dropout),
      nn.Linear[ParamType](final_fc_dim, 256),
      nn.ReLU[ParamType](),
      nn.Dropout[ParamType](dropout),
      nn.Linear[ParamType](256, num_skills)
    )
  } else {
    nn.Sequential[ParamType](
      nn.Linear[ParamType](embedding_size + embed_l, final_fc_dim),
      nn.ReLU[ParamType](),
      nn.Dropout[ParamType](dropout),
      nn.Linear[ParamType](final_fc_dim, 256),
      nn.ReLU[ParamType](),
      nn.Dropout[ParamType](dropout),
      nn.Linear[ParamType](256, 1)
    )
  }

  // 损失函数
  val loss_fn = nn.BCELoss(reduction = "mean")

  // 初始化参数
  reset()

  def reset(): Unit = {
    for (p <- this.parameters()) {
      if (num_questions > 0 && p.data.shape(0) == num_questions + 1) {
        nn.init.constant_(p.data, 0.0)
      }
    }
  }

  def base_emb(q_data: Tensor[ParamType], target: Tensor[ParamType]): (Tensor[ParamType], Tensor[ParamType]) = {
    val q_embed_data = q_embed.get.apply(q_data)  // BS, seqlen, embedding_size
    val qa_embed_data = if (separate_qr) {
      val qa_data = q_data + num_skills * target
      qa_embed.get.apply(qa_data)
    } else {
      // BS, seqlen, embedding_size # c_ct + g_rt = e_(ct,rt)
      qa_embed.get.apply(target) + q_embed_data
    }
    (q_embed_data, qa_embed_data)
  }

  def forward(feed_dict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = {
    val pid_data = feed_dict("questions")
    val r = feed_dict("responses")
    val c = feed_dict("skills")
    var attention_mask = feed_dict("attention_mask")
    var q_data = c
    val target = r * (r > -1)

    var cshft: Option[Tensor[ParamType]] = None

    // 根据不同模式处理数据
    if (trans) {
      val pid_data_new = pid_data.slice(1, 0, pid_data.shape(1) - length)
      val q_data_new = q_data.slice(1, 0, q_data.shape(1) - length)
      val target_new = target.slice(1, 0, target.shape(1) - length)
      val attention_mask_new = attention_mask.slice(1, 0, attention_mask.shape(1) - length)
      cshft = Some(c.slice(1, length, c.shape(1)))
      q_data = q_data_new
    } else if (mask_future) {
      val mask_part = attention_mask.slice(1, attention_mask.shape(1) - length, attention_mask.shape(1))
      attention_mask = attention_mask.slice(1, 0, attention_mask.shape(1) - length).cat(torch.zeros_like(mask_part), dim = 1)
      pid_data = pid_data * attention_mask
      q_data = q_data * attention_mask
      target = target * attention_mask
    } else if (mask_response) {
      val mask_part = attention_mask.slice(1, attention_mask.shape(1) - length, attention_mask.shape(1))
      attention_mask = attention_mask.slice(1, 0, attention_mask.shape(1) - length).cat(torch.zeros_like(mask_part), dim = 1)
      target = target * attention_mask
    }

    // 获取嵌入
    val (q_embed_data, qa_embed_data) = base_emb(q_data, target)

    // 处理问题难度参数
    var pid_embed_data: Option[Tensor[ParamType]] = None
    var c_reg_loss: Tensor[ParamType] = torch.zeros()

    if (num_questions > 0) {
      val q_embed_diff_data = q_embed_diff.get.apply(q_data)
      pid_embed_data = Some(difficult_param.get.apply(pid_data))
      q_embed_data := q_embed_data + pid_embed_data.get * q_embed_diff_data

      val qa_embed_diff_data = qa_embed_diff.get.apply(target)
      if (separate_qr) {
        qa_embed_data := qa_embed_data + pid_embed_data.get * qa_embed_diff_data
      } else {
        qa_embed_data := qa_embed_data + pid_embed_data.get * (qa_embed_diff_data + q_embed_diff_data)
      }

      c_reg_loss = (pid_embed_data.get ** 2.0).sum() * l2
    }

    // 模型前向传播
    val d_output = model(q_embed_data, qa_embed_data, pid_embed_data)

    // 计算池化分数
    val pooled_ques_score = (q_embed.get.apply(q_data) * attention_mask.unsqueeze(-1)).sum(1) / 
      attention_mask.sum(-1).unsqueeze(-1)
    val pooled_inter_score = (qa_embed_data * attention_mask.unsqueeze(-1)).sum(1) / 
      attention_mask.sum(-1).unsqueeze(-1)

    // 连接并通过输出层
    val concat_q = torch.cat(Seq(d_output, q_embed_data), dim = -1)
    val output = if (trans) {
      out(concat_q)
    } else {
      out(concat_q).squeeze(-1)
    }
    val output_sigmoid = F.sigmoid(output)

    // 根据模式处理输出
    val (final_output, true_output) = if (trans) {
      val cshft_tensor = cshft.get
      val one_hot = torch.zeros_like(output_sigmoid)
      // 实现one_hot编码
      for (i <- 0 until cshft_tensor.shape(0).int()) {
        for (j <- 0 until cshft_tensor.shape(1).int()) {
          val idx = cshft_tensor(i)(j).int()
          if (idx >= 0 && idx < output_sigmoid.shape(2).int()) {
            one_hot(i)(j)(idx) = torch.ones(1)
          }
        }
      }
      val out = (output_sigmoid * one_hot).sum(-1)
      val true_out = r.slice(1, length, r.shape(1))
      (out, true_out)
    } else if (mask_future || pred_last || mask_response) {
      val out = output_sigmoid.slice(1, output_sigmoid.shape(1) - length, output_sigmoid.shape(1))
      val true_out = r.slice(1, r.shape(1) - length, r.shape(1))
      (out, true_out)
    } else {
      val out = output_sigmoid.slice(1, length, output_sigmoid.shape(1))
      val true_out = r.slice(1, length, r.shape(1))
      (out, true_out)
    }

    // 返回结果
    if (this.isTraining) {
      Map(
        "pred" -> final_output,
        "true" -> true_output,
        "c_reg_loss" -> c_reg_loss
      )
    } else {
      Map(
        "pred" -> final_output,
        "true" -> true_output,
        "c_reg_loss" -> c_reg_loss,
        "q_embed" -> pooled_ques_score,
        "qr_embed" -> pooled_inter_score
      )
    }
  }

  def loss(feed_dict: Map[String, Tensor[ParamType]], out_dict: Map[String, Tensor[ParamType]]): (Tensor[ParamType], Int, Double) = {
    val pred = out_dict("pred").flatten()
    val true_values = out_dict("true").flatten()
    val c_reg_loss = out_dict("c_reg_loss")
    val mask = true_values > -1
    val loss = loss_fn(pred.masked_select(mask), true_values.masked_select(mask))
    val total_loss = loss + c_reg_loss
    val count = pred.masked_select(mask).shape(0)//.int()
    val sum_true = true_values.masked_select(mask).sum().item().double()
    (total_loss, count, sum_true)
  }

  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    throw new NotImplementedError("This model expects a Map input")
  }
}

// 架构类
class Architecture4[ParamType <: FloatNN: Default] 
    (num_skills: Int, 
     num_blocks: Int, 
     embedding_size: Int, 
     d_feature: Double, 
     d_ff: Int, 
     n_heads: Int, 
     dropout: Double, 
     kq_same: Boolean, 
     model_type: String, 
     seq_len: Int, 
     emb_type: String, 
     num_buckets: Int, 
     max_distance: Int)
    extends TensorModule[ParamType] 
    with HasParams[ParamType] {

  val position_emb: Option[SinePositionalEncoding[ParamType]] = 
    if (emb_type.contains("sin")) Some(new SinePositionalEncoding[ParamType](embedding_size, seq_len)) else None

  // 创建Transformer层
  val blocks_1 = ListBuffer[TransformerLayer[ParamType]]()
  val blocks_2 = ListBuffer[TransformerLayer[ParamType]]()

  if (model_type == "folibikt") {
    for (_ <- 0 until num_blocks) {
      blocks_1 += new TransformerLayer4[ParamType](
        embedding_size = embedding_size,
        d_feature = embedding_size / n_heads,
        d_ff = d_ff,
        n_heads = n_heads,
        dropout = dropout,
        kq_same = kq_same,
        emb_type = emb_type,
        num_buckets = num_buckets,
        max_distance = max_distance
      )
    }

    for (_ <- 0 until num_blocks * 2) {
      blocks_2 += new TransformerLayer4[ParamType](
        embedding_size = embedding_size,
        d_feature = embedding_size / n_heads,
        d_ff = d_ff,
        n_heads = n_heads,
        dropout = dropout,
        kq_same = kq_same,
        emb_type = emb_type,
        num_buckets = num_buckets,
        max_distance = max_distance
      )
    }
  }

  def forward(q_embed_data: Tensor[ParamType], 
              qa_embed_data: Tensor[ParamType], 
              pid_embed_data: Option[Tensor[ParamType]]): Tensor[ParamType] = {
    // 添加位置编码
    var q_embed_with_pos = q_embed_data
    var qa_embed_with_pos = qa_embed_data

    if (position_emb.isDefined) {
      q_embed_with_pos = q_embed_with_pos + position_emb.get(q_embed_with_pos)
      qa_embed_with_pos = qa_embed_with_pos + position_emb.get(qa_embed_with_pos)
    }

    var y = qa_embed_with_pos
    var x = q_embed_with_pos

    // 编码器部分
    for (block <- blocks_1) {
      y = block(mask = 1, query = y, key = y, values = y, pdiff = pid_embed_data)
    }

    // 处理第二个块序列
    var flag_first = true
    for (block <- blocks_2) {
      if (flag_first) {
        x = block(mask = 1, query = x, key = x, values = x, apply_pos = false, pdiff = pid_embed_data)
        flag_first = false
      } else {
        x = block(mask = 0, query = x, key = x, values = y, apply_pos = true, pdiff = pid_embed_data)
        flag_first = true
      }
    }

    x
  }

  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    throw new NotImplementedError("This model expects multiple inputs")
  }
}

// Transformer层
class TransformerLayer4[ParamType <: FloatNN: Default] 
    (embedding_size: Int, 
     d_feature: Double, 
     d_ff: Int, 
     n_heads: Int, 
     dropout: Double, 
     kq_same: Boolean, 
     emb_type: String, 
     num_buckets: Int, 
     max_distance: Int)
    extends TensorModule[ParamType] 
    with HasParams[ParamType] {

  // 多头注意力层
  val masked_attn_head = new MultiHeadAttention4[ParamType](
    embedding_size = embedding_size,
    d_feature = d_feature.toInt,
    n_heads = n_heads,
    dropout = dropout,
    kq_same = kq_same,
    num_buckets = num_buckets,
    max_distance = max_distance,
    emb_type = emb_type
  )

  // 层归一化和Dropout
  val layer_norm1 = nn.LayerNorm[ParamType](embedding_size)
  val dropout1 = nn.Dropout[ParamType](dropout)

  // 前馈网络
  val linear1 = nn.Linear[ParamType](embedding_size, d_ff)
  val activation = nn.ReLU[ParamType]()
  val dropout = nn.Dropout[ParamType](dropout)
  val linear2 = nn.Linear[ParamType](d_ff, embedding_size)

  val layer_norm2 = nn.LayerNorm[ParamType](embedding_size)
  val dropout2 = nn.Dropout[ParamType](dropout)

  def forward(mask: Int, 
              query: Tensor[ParamType], 
              key: Tensor[ParamType], 
              values: Tensor[ParamType], 
              apply_pos: Boolean = true, 
              pdiff: Option[Tensor[ParamType]] = None): Tensor[ParamType] = {
    val seqlen = query.shape(1)
    val batch_size = query.shape(0)
    val device = query.device

    // 创建掩码
    val nopeek_mask = torch.ones(1, 1, seqlen, seqlen)
    for (i <- 0 until seqlen; j <- 0 until seqlen) {
      if (j >= i + mask) {
        nopeek_mask(0)(0)(i)(j) = torch.zeros(1)
      }
    }
    val src_mask = nopeek_mask.to(device)

    // 计算自注意力
    val query2 = if (mask == 0) {
      masked_attn_head(query, key, values, mask = src_mask, zero_pad = true, pdiff = pdiff)
    } else {
      masked_attn_head(query, key, values, mask = src_mask, zero_pad = false, pdiff = pdiff)
    }

    // 残差连接和层归一化
    var result = query + dropout1(query2)
    result = layer_norm1(result)

    // 前馈网络
    if (apply_pos) {
      val ffn_output = linear2(dropout(activation(linear1(result))))
      result = result + dropout2(ffn_output)
      result = layer_norm2(result)
    }

    result
  }

  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    throw new NotImplementedError("This layer expects multiple inputs")
  }
}

// 多头注意力
class MultiHeadAttention4[ParamType <: FloatNN: Default] 
    (embedding_size: Int, 
     d_feature: Int, 
     n_heads: Int, 
     dropout: Double, 
     kq_same: Boolean, 
     num_buckets: Int, 
     max_distance: Int, 
     bias: Boolean = true, 
     emb_type: String = "qid")
    extends TensorModule[ParamType] 
    with HasParams[ParamType] {

  val d_k = d_feature
  val h = n_heads

  // 根据嵌入类型初始化不同的组件
  val rel_pos_bias: Option[T5RelativePositionBias[ParamType]] = 
    if (emb_type.contains("t5")) Some(new T5RelativePositionBias[ParamType](math.sqrt(embedding_size), true, num_buckets, max_distance)) else None

  val rotary_pe: Option[RotaryPositionalEmbeddings[ParamType]] = 
    if (emb_type.contains("rotary")) Some(new RotaryPositionalEmbeddings[ParamType](d_k)) else None

  // 线性层
  var v_linear: Option[nn.Linear[ParamType]] = None
  var k_linear: Option[nn.Linear[ParamType]] = None
  var q_linear: Option[nn.Linear[ParamType]] = None
  var linear: Option[nn.Linear[ParamType]] = None
  var pooling: Option[nn.AvgPool1d[ParamType]] = None

  val out_proj = nn.Linear[ParamType](embedding_size, embedding_size, bias = bias)
  val proj_bias = bias

  // 根据嵌入类型初始化不同的层
  if (emb_type.endsWith("avgpool")) {
    val pool_size = 3
    pooling = Some(nn.AvgPool1d[ParamType](pool_size, stride = 1, padding = pool_size / 2, count_include_pad = false))
  } else if (emb_type.endsWith("linear")) {
    linear = Some(nn.Linear[ParamType](embedding_size, embedding_size, bias = bias))
  } else if (emb_type.startsWith("qid")) {
    v_linear = Some(nn.Linear[ParamType](embedding_size, embedding_size, bias = bias))
    k_linear = Some(nn.Linear[ParamType](embedding_size, embedding_size, bias = bias))
    if (!kq_same) {
      q_linear = Some(nn.Linear[ParamType](embedding_size, embedding_size, bias = bias))
    }
  }

  // gamma参数
  val gammas = torch.zeros(n_heads, 1, 1).requiresGrad_(true)
  nn.init.xavier_uniform_(gammas)

  // 初始化参数
  _reset_parameters()

  // 计算slopes和alibi
  private def get_slopes(n: Int): Array[Double] = {
    def get_slopes_power_of_2(n: Int): Array[Double] = {
      val start = math.pow(2, -math.pow(2, -(math.log(n) / math.log(2) - 3)))
      var ratio = start
      (0 until n).map(i => start * math.pow(ratio, i)).toArray
    }

    if (math.log(n) / math.log(2) % 1 == 0) {
      get_slopes_power_of_2(n)
    } else {
      val closest_power_of_2 = math.pow(2, math.floor(math.log(n) / math.log(2))).int()
      val slopes_power_of_2 = get_slopes_power_of_2(closest_power_of_2)
      val slopes_double = get_slopes_power_of_2(2 * closest_power_of_2)
      val slopes_remaining = slopes_double.slice(0, 2 * closest_power_of_2 - closest_power_of_2).grouped(2).map(_.head).toArray
      slopes_power_of_2 ++ slopes_remaining.slice(0, n - closest_power_of_2)
    }
  }

  // 计算alibi
  val maxpos = 1000
  val attn_heads = n_heads
  val context_position = torch.arange(maxpos).unsqueeze(1)
  val memory_position = torch.arange(maxpos).unsqueeze(0)
  val relative_position = (memory_position - context_position).abs().unsqueeze(0).expand(Seq(attn_heads, -1, -1))

  val slopes = Tensor.fromArray(get_slopes(attn_heads)) * -1
  val alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position
  val alibi_view = alibi.view(1, attn_heads, maxpos, maxpos)

  private def _reset_parameters(): Unit = {
    if (emb_type.startsWith("qid")) {
      nn.init.xavier_uniform_(k_linear.get.weight)
      nn.init.xavier_uniform_(v_linear.get.weight)
      if (!kq_same) {
        nn.init.xavier_uniform_(q_linear.get.weight)
      }

      if (proj_bias) {
        nn.init.constant_(k_linear.get.bias, 0.0)
        nn.init.constant_(v_linear.get.bias, 0.0)
        if (!kq_same) {
          nn.init.constant_(q_linear.get.bias, 0.0)
        }
        nn.init.constant_(out_proj.bias, 0.0)
      }
    }
  }

  def forward(q: Tensor[ParamType], 
              k: Tensor[ParamType], 
              v: Tensor[ParamType], 
              mask: Tensor[ParamType], 
              zero_pad: Boolean, 
              pdiff: Option[Tensor[ParamType]] = None): Tensor[ParamType] = {

    val bs = q.shape(0).int()

    val concat = if (emb_type.endsWith("avgpool")) {
      val scores = pooling.get.apply(v.transpose(1, 2)).transpose(1, 2)
      pad_zero(scores, bs, scores.shape(2).int(), zero_pad)
    } else if (emb_type.endsWith("linear")) {
      val scores = linear.get.apply(v)
      pad_zero(scores, bs, scores.shape(2).int(), zero_pad)
    } else if (emb_type.startsWith("qid")) {
      // 线性变换并分割成多个头
      val k_transformed = k_linear.get.apply(k).view(bs, -1, h, d_k)
      val q_transformed = if (!kq_same) {
        q_linear.get.apply(q).view(bs, -1, h, d_k)
      } else {
        k_linear.get.apply(q).view(bs, -1, h, d_k)
      }
      val v_transformed = v_linear.get.apply(v).view(bs, -1, h, d_k)

      // 转置以获得维度 bs * h * sl * embedding_size
      val k_transposed = k_transformed.transpose(1, 2)
      val q_transposed = q_transformed.transpose(1, 2)
      val v_transposed = v_transformed.transpose(1, 2)

      // 计算注意力
      val adjusted_pdiff = if (emb_type.contains("pdiff")) pdiff else None
      val scores = attention(
        q_transposed, 
        k_transposed, 
        v_transposed, 
        d_k.double(),
        mask, 
        nn.Dropout[ParamType](dropout), 
        zero_pad, 
        Some(gammas), 
        adjusted_pdiff, 
        Some(alibi_view), 
        emb_type, 
        rel_pos_bias, 
        rotary_pe
      )

      // 连接头并通过最终线性层
      scores.transpose(1, 2).contiguous().view(bs, -1, embedding_size)
    } else {
      throw new IllegalArgumentException(s"Unsupported embedding type: $emb_type")
    }

    // 输出投影
    out_proj(concat)
  }

  def pad_zero(scores: Tensor[ParamType], bs: Int, dim: Int, zero_pad: Boolean): Tensor[ParamType] = {
    if (zero_pad) {
      val device = scores.device
      val pad_zero = torch.zeros(bs, 1, dim).to(device)
      torch.cat(Seq(pad_zero, scores.slice(1, 0, scores.shape(1) - 1)), dim = 1)
    } else {
      scores
    }
  }

  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    throw new NotImplementedError("This layer expects multiple inputs")
  }
}

// 注意力函数
object attention {
  def apply[ParamType <: FloatNN: Default](
      q: Tensor[ParamType], 
      k: Tensor[ParamType], 
      v: Tensor[ParamType], 
      d_k: Double, 
      mask: Tensor[ParamType], 
      dropout: nn.Dropout[ParamType], 
      zero_pad: Boolean, 
      gamma: Option[Tensor[ParamType]] = None, 
      pdiff: Option[Tensor[ParamType]] = None, 
      alibi: Option[Tensor[ParamType]] = None, 
      emb_type: String = "qid", 
      rel_pos_bias: Option[T5RelativePositionBias[ParamType]] = None, 
      rotary_pe: Option[RotaryPositionalEmbeddings[ParamType]] = None
  ): Tensor[ParamType] = {

    // 计算注意力分数
    var scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    val bs = scores.shape(0)
    val head = scores.shape(1)
    val seqlen = scores.shape(2)
    val device = q.device

    // 创建位置索引
    val x1 = torch.arange(end = seqlen).expand(Seq(seqlen, -1)).to(device)
    val x2 = x1.transpose(0, 1).contiguous()

    // 处理不同类型的注意力
    if (emb_type.contains("alibi") && alibi.isDefined) {
      val seq_len = scores.shape(3)
      scores = scores + alibi.get.slice(2, 0, seq_len).slice(3, 0, seq_len)
    }

    // 计算距离效果
    val dist_scores = {
      val scores_ = scores.masked_fill(mask == 0, -1e32f)
      val softmax_scores = F.softmax(scores_, dim = -1)
      val masked_softmax = softmax_scores * mask
      val distcum_scores = masked_softmax.cumsum(dim = -1)
      val disttotal_scores = masked_softmax.sum(dim = -1, keepdim = true)
      val position_effect = (x1 - x2).abs().unsqueeze(0).unsqueeze(0)
      val dist_scores_unclamped = (disttotal_scores - distcum_scores) * position_effect
      dist_scores_unclamped.clamp(min = 0.0).sqrt().detach()
    }

    // 计算总效果
    val total_effect = {
      val m = nn.Softplus[ParamType]()
      val gamma_value = -1.0 * m(gamma.get).unsqueeze(0)
      if (pdiff.isEmpty) {
        dist_scores * gamma_value
      } else {
        val diff = pdiff.get.unsqueeze(1).expand(
          Seq(pdiff.get.shape(0), dist_scores.shape(1), pdiff.get.shape(1), pdiff.get.shape(2))
        )
        val diff_sigmoid = F.sigmoid(diff).exp()
        dist_scores * gamma_value * diff_sigmoid
      }
    }.exp().clamp(min = 1e-5f, max = 1e5f)

    // 应用总效果
    scores = scores * total_effect

    // 应用掩码和softmax
    scores = scores.masked_fill(mask == 0, -1e32f)
    scores = F.softmax(scores, dim = -1)

    // 零填充
    if (zero_pad) {
      val pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
      scores = torch.cat(Seq(pad_zero, scores.slice(2, 1, scores.shape(2))), dim = 2)
    }

    // 应用dropout并计算输出
    scores = dropout(scores)
    torch.matmul(scores, v)
  }
}

// 正弦位置编码
class SinePositionalEncoding[ParamType <: FloatNN: Default](d_hid: Int, n_position: Int = 200)
    extends TensorModule[ParamType] 
    with HasParams[ParamType] {

  // 注册位置编码表
  val pos_table = _get_sinusoid_encoding_table(1000, d_hid)

  private def _get_sinusoid_encoding_table(n_position: Int, d_hid: Int): Tensor[ParamType] = {
    // 创建正弦位置编码表
    val sinusoid_table = torch.zeros[ParamType](n_position, d_hid)
    val position = torch.arange(end = n_position).unsqueeze(1)
    val div_term = torch.exp(
      torch.arange(0, d_hid, 2) *
      -(math.log(10000.0) / d_hid)
    )
    
    // 奇数维度使用sin，偶数维度使用cos
    for (i <- 0 until n_position) {
      for (j <- 0 until d_hid by 2) {
        if (j < d_hid) {
          sinusoid_table(i)(j) = torch.sin(position(i)(0) * div_term(j / 2))
        }
        if (j + 1 < d_hid) {
          sinusoid_table(i)(j + 1) = torch.cos(position(i)(0) * div_term(j / 2))
        }
      }
    }
    
    sinusoid_table.unsqueeze(0)
  }

  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    pos_table.slice(1, 0, x.shape(1))
  }

  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)
}

// T5相对位置偏置
class T5RelativePositionBias[ParamType <: FloatNN: Default](
    scale: Double, 
    causal: Boolean = true, 
    num_buckets: Int = 16, 
    max_distance: Int = 50
) extends TensorModule[ParamType] 
    with HasParams[ParamType] {

  val relative_attention_bias = nn.Embedding[ParamType](num_buckets, 1)

  def _relative_position_bucket(
      relative_position: Tensor[ParamType],
      causal: Boolean = true,
      num_buckets: Int = 16,
      max_distance: Int = 50
  ): Tensor[ParamType] = {
    var ret = torch.zeros_like(relative_position)
    val n = -relative_position

    if (!causal) {
      val half_buckets = num_buckets / 2
      ret += (n < 0) * half_buckets
      n := n.abs()
    } else {
      n := torch.max(n, torch.zeros_like(n))
    }

    val max_exact = num_buckets / 2
    val is_small = n < max_exact

    val val_if_large = max_exact + (
      torch.log(n / max_exact) /
      math.log(max_distance.toDouble) / max_exact) *
      (num_buckets - max_exact)
    )
    val val_if_large_clamped = torch.min(
      val_if_large, 
      torch.full_like(val_if_large, num_buckets - 1)
    )

    ret += torch.where(is_small, n, val_if_large_clamped)
    ret
  }

  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    val i = x.shape(x.dim - 2)
    val j = x.shape(x.dim - 1)
    val device = x.device
    
    val q_pos = torch.arange(end = i, device = device)
    val k_pos = torch.arange(end =j, device = device)
    
    // 计算相对位置
    val rel_pos = k_pos.unsqueeze(0) - q_pos.unsqueeze(1)
    
    // 获取位置桶
    val rp_bucket = _relative_position_bucket(
      rel_pos,
      causal = causal,
      num_buckets = num_buckets,
      max_distance = max_distance
    )
    
    // 获取偏置值
    val values = relative_attention_bias(rp_bucket)
    
    // 调整形状并应用缩放
    values.squeeze(-1) * scale
  }

  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)
}

// 旋转位置编码
class RotaryPositionalEmbeddings[ParamType <: FloatNN: Default](d: Int, base: Int = 10000)
    extends TensorModule[ParamType] 
    with HasParams[ParamType] {

  // 计算θ参数
  val theta = Tensor.fromArray(
    (0 until d by 2).map(i => 1.0 / math.pow(base, i.double() / d)).toArray
  ).requiresGrad_(false)

  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    // 提取形状
    val batch_size = x.shape(0)
    val seq_len = x.shape(1)
    val n_heads = x.shape(2)
    val d = x.shape(3)

    // d/2
    val d_2 = d / 2

    // 创建位置索引
    val seq_idx = torch.arange(seq_len)

    // 计算位置索引和θ的乘积
    val idx_theta = torch.einsum("n,d->nd", Seq(seq_idx, theta))

    // 连接以便每行m有[mθ0, mθ1, ..., mθd/2, mθ0, mθ1, ..., mθd/2]
    val idx_theta2 = torch.cat(Seq(idx_theta, idx_theta), dim = 1)

    // 计算[-x^(d/2+1), -x^(d/2+2), ..., -x^d, x^1, x^2, ..., x^(d/2)]
    val neg_half_x = torch.cat(
      Seq(-x.slice(3, d_2, d), x.slice(3, 0, d_2)), 
      dim = -1
    )

    // 计算旋转位置编码
    val rx = (x * idx_theta2.cos().unsqueeze(0).unsqueeze(2)) + 
             (neg_half_x * idx_theta2.sin().unsqueeze(0).unsqueeze(2))

    rx
  }

  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)
}

// 伴生对象提供工厂方法
object folibiKT {
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
      d_ff: Int = 256, 
      kq_same: Boolean = true, 
      final_fc_dim: Int = 512, 
      num_attn_heads: Int = 8, 
      separate_qr: Boolean = false, 
      l2: Double = 1e-5, 
      emb_type: String = "qid_alibi", 
      num_buckets: Int = 32, 
      max_distance: Int = 100
  ): folibiKT[ParamType] = {
    new folibiKT[ParamType](
      mask_response, pred_last, mask_future, length, trans, num_skills, num_questions, 
      seq_len, embedding_size, num_blocks, dropout, d_ff, kq_same, final_fc_dim, 
      num_attn_heads, separate_qr, l2, emb_type, num_buckets, max_distance
    )
  }
}
