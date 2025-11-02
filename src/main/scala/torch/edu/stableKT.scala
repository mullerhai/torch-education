package torch.edu

import torch.*
import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn as nn
import torch.nn.*
import scala.collection.mutable.ListBuffer

object Dim extends Enumeration {
  type Dim = Value
  val batch, seq, feature = Value
}

class stableKT[ParamType <: FloatNN: Default](
    mask_response: Boolean,
    pred_last: Boolean,
    mask_future: Boolean,
    length: Int,
    trans: Boolean,
    num_skills: Int,
    num_questions: Int,
    embedding_size: Int,
    num_blocks: Int,
    dropout: Float,
    d_ff: Int = 256,
    seq_len: Int = 200,
    kq_same: Int = 1,
    final_fc_dim: Int = 512,
    final_fc_dim2: Int = 256,
    num_attn_heads: Int = 8,
    separate_qr: Boolean = false,
    l2: Float = 1e-5f,
    r: Float = 1.0f,
    gamma: Float = 1.0f,
    num_buckets: Int = 32,
    max_distance: Int = 100
) extends HasParams[ParamType] with TensorModule[ParamType] {
  
  val hidden_size: Int = embedding_size
  
  // Embedding layers
  val difficult_param = if (num_questions > 0) {
    Some(nn.Embedding[ParamType](num_questions + 1, embedding_size))
  } else None
  
  val q_embed_diff = if (num_questions > 0) {
    Some(nn.Embedding[ParamType](num_skills + 1, embedding_size))
  } else None
  
  val qa_embed_diff = if (num_questions > 0) {
    Some(nn.Embedding[ParamType](2 * num_skills + 1, embedding_size))
  } else None
  
  val q_embed = nn.Embedding[ParamType](num_skills, embedding_size)
  
  val qa_embed = if (separate_qr) {
    nn.Embedding[ParamType](2 * num_skills + 1, embedding_size)
  } else {
    nn.Embedding[ParamType](2, embedding_size)
  }
  
  // Architecture Object
  val model = new Architecture2[ParamType](
    num_skills = num_skills,
    num_blocks = num_blocks,
    n_heads = num_attn_heads,
    dropout = dropout,
    d_model = hidden_size,
    d_feature = hidden_size / num_attn_heads,
    d_ff = d_ff,
    kq_same = kq_same,
    seq_len = seq_len,
    r = r,
    gamma = gamma,
    num_buckets = num_buckets,
    max_distance = max_distance
  )
  
  // Output Layer
  val out = if (trans) {
    nn.Sequential[ParamType](
      nn.Linear[ParamType](hidden_size + embedding_size, final_fc_dim),
      nn.ReLU[ParamType](),
      nn.Dropout[ParamType](dropout),
      nn.Linear[ParamType](final_fc_dim, final_fc_dim2),
      nn.ReLU[ParamType](),
      nn.Dropout[ParamType](dropout),
      nn.Linear[ParamType](final_fc_dim2, num_skills)
    )
  } else {
    nn.Sequential[ParamType](
      nn.Linear[ParamType](hidden_size + embedding_size, final_fc_dim),
      nn.ReLU[ParamType](),
      nn.Dropout[ParamType](dropout),
      nn.Linear[ParamType](final_fc_dim, final_fc_dim2),
      nn.ReLU[ParamType](),
      nn.Dropout[ParamType](dropout),
      nn.Linear[ParamType](final_fc_dim2, 1)
    )
  }
  
  val loss_fn = nn.BCELoss()
  
  reset()
  
  def reset(): Unit = {
    if (num_questions > 0) {
      difficult_param.foreach { param =>
        param.weight().data().fill_(0.0f)
      }
    }
  }
  
  def base_emb(q_data: Tensor[ParamType], target: Tensor[ParamType]): (Tensor[ParamType], Tensor[ParamType]) = {
    val q_embed_data = q_embed(q_data)  // BS, seqlen, d_model
    
    val qa_embed_data = if (separate_qr) {
      val qa_data = q_data + (target * num_skills.toLong).long()
      qa_embed(qa_data)
    } else {
      qa_embed(target) + q_embed_data
    }
    
    (q_embed_data, qa_embed_data)
  }
  
  def apply(input: Map[String, Tensor[ParamType]], qtest: Boolean = false, train: Boolean = false): Map[String, Tensor[ParamType]] = {
    forward(input, qtest, train)
  }
  
  def forward(feed_dict: Map[String, Tensor[ParamType]], qtest: Boolean = false, train: Boolean = false): Map[String, Tensor[ParamType]] = {
    val q = feed_dict("questions")
    val c = feed_dict("skills")
    val r = feed_dict("responses")
    val masked_r = r * (r > (-1.0f)).long()
    val attention_mask = feed_dict("attention_mask")
    
    var cshft: Tensor[ParamType] = null
    var processed_q = q
    var processed_c = c
    var processed_masked_r = masked_r
    var processed_attention_mask = attention_mask
    
    if (trans) {
      cshft = c.slice(1, length, c.size(1))
      processed_q = q.slice(1, 0, q.size(1) - length)
      processed_c = c.slice(1, 0, c.size(1) - length)
      processed_masked_r = masked_r.slice(1, 0, masked_r.size(1) - length)
      processed_attention_mask = attention_mask.slice(1, 0, attention_mask.size(1) - length)
    } else if (mask_future) {
      processed_attention_mask = processed_attention_mask.masked_fill(processed_attention_mask.size(1) - length until processed_attention_mask.size(1), 0.0f)
      processed_q = processed_q * processed_attention_mask
      processed_c = processed_c * processed_attention_mask
      processed_masked_r = processed_masked_r * processed_attention_mask
    } else if (mask_response) {
      processed_attention_mask = processed_attention_mask.masked_fill(processed_attention_mask.size(1) - length until processed_attention_mask.size(1), 0.0f)
      processed_masked_r = processed_masked_r * processed_attention_mask
    }
    
    val (q_embed_data, qa_embed_data) = base_emb(processed_c, processed_masked_r)
    
    val final_q_embed_data = if (num_questions > 0) {
      val q_embed_diff_data = q_embed_diff.get(processed_c)
      val pid_embed_data = difficult_param.get(processed_q)
      q_embed_data + (pid_embed_data * q_embed_diff_data)
    } else {
      q_embed_data
    }
    
    val d_output = model(final_q_embed_data, qa_embed_data)
    
    val concat_q = torch.cat(Seq(d_output, final_q_embed_data), dim = -1)
    
    val output = if (trans) {
      out(concat_q)
    } else {
      out(concat_q).squeeze(-1)
    }
    
    val m = nn.Sigmoid[ParamType]()
    val output_sigmoid = m(output)
    
    val (final_output, true_tensor) = if (trans) {
      val one_hot_cshft = F.one_hot(cshft.long(), num_skills)
      val out = (output_sigmoid * one_hot_cshft).sum(-1)
      val true_val = r.slice(1, length, r.size(1))
      (out, true_val)
    } else if (mask_future || pred_last || mask_response) {
      val out = output_sigmoid.slice(1, output_sigmoid.size(1) - length, output_sigmoid.size(1))
      val true_val = r.slice(1, r.size(1) - length, r.size(1))
      (out, true_val)
    } else {
      val out = output_sigmoid.slice(1, length, output_sigmoid.size(1))
      val true_val = r.slice(1, length, r.size(1))
      (out, true_val)
    }
    
    Map(
      "pred" -> final_output,
      "true" -> true_tensor
    )
  }
  
  def loss(feed_dict: Map[String, Tensor[ParamType]], out_dict: Map[String, Tensor[ParamType]]): (Tensor[ParamType], Long, Double) = {
    val pred = out_dict("pred").flatten()
    val true_tensor = out_dict("true").flatten()
    val mask = true_tensor > (-1.0f)
    val loss = loss_fn(pred.masked_select(mask), true_tensor.masked_select(mask))
    (loss, mask.sum().long(), true_tensor.masked_select(mask).sum().item().double)
  }
  
  override def parameters: Seq[Parameter[ParamType]] = {
    val params = ListBuffer[Parameter[ParamType]]()
    params ++= q_embed.parameters
    params ++= qa_embed.parameters
    difficult_param.foreach(params ++= _.parameters)
    q_embed_diff.foreach(params ++= _.parameters)
    qa_embed_diff.foreach(params ++= _.parameters)
    params ++= model.parameters
    params ++= out.parameters
    params.toSeq
  }
}

class Architecture7[ParamType <: FloatNN: Default](
    num_skills: Int,
    num_blocks: Int,
    d_model: Int,
    d_feature: Double,
    d_ff: Int,
    n_heads: Int,
    dropout: Float,
    kq_same: Int,
    seq_len: Int,
    r: Float,
    gamma: Float,
    num_buckets: Int,
    max_distance: Int
) extends HasParams[ParamType] with TensorModule[ParamType] {
  
  val blocks_2 = ListBuffer[TransformerLayer2[ParamType]]()
  for (_ <- 0 until num_blocks) {
    blocks_2 += new TransformerLayer2[ParamType](
      d_model = d_model,
      d_feature = d_model / n_heads,
      d_ff = d_ff,
      dropout = dropout,
      n_heads = n_heads,
      kq_same = kq_same,
      seq_len = seq_len,
      r = r,
      gamma = gamma,
      num_buckets = num_buckets,
      max_distance = max_distance
    )
  }
  
  val position_emb = new CosinePositionalEmbedding2[ParamType](d_model = d_model, max_len = seq_len)
  
  def apply(q_embed_data: Tensor[ParamType], qa_embed_data: Tensor[ParamType]): Tensor[ParamType] = {
    forward(q_embed_data, qa_embed_data)
  }
  
  def forward(q_embed_data: Tensor[ParamType], qa_embed_data: Tensor[ParamType]): Tensor[ParamType] = {
    val seqlen = q_embed_data.size(1)
    val batch_size = q_embed_data.size(0)
    
    val qa_pos_embed = qa_embed_data
    val q_pos_embed = q_embed_data
    
    var y = qa_pos_embed
    var x = q_pos_embed
    
    // encoder
    for (block <- blocks_2) {
      x = block(mask = 0, query = x, key = x, values = y, apply_pos = true)
    }
    
    x
  }
  
  override def parameters: Seq[Parameter[ParamType]] = {
    val params = ListBuffer[Parameter[ParamType]]()
    blocks_2.foreach(params ++= _.parameters)
    params ++= position_emb.parameters
    params.toSeq
  }
}

class TransformerLayer2[ParamType <: FloatNN: Default](
    d_model: Int,
    d_feature: Int,
    d_ff: Int,
    n_heads: Int,
    dropout: Float,
    kq_same: Int,
    seq_len: Int,
    r: Float,
    gamma: Float,
    num_buckets: Int,
    max_distance: Int
) extends HasParams[ParamType] with TensorModule[ParamType] {
  
  val masked_attn_head = new MultiHeadAttention2[ParamType](
    d_model = d_model,
    d_feature = d_feature,
    n_heads = n_heads,
    dropout = dropout,
    kq_same = kq_same == 1,
    seq_len = seq_len,
    r = r,
    gamma = gamma,
    num_buckets = num_buckets,
    max_distance = max_distance
  )
  
  val layer_norm1 = nn.LayerNorm[ParamType](d_model)
  val dropout1 = nn.Dropout[ParamType](dropout)
  
  val linear1 = nn.Linear[ParamType](d_model, d_ff)
  val activation = nn.ReLU[ParamType]()
  val dropout = nn.Dropout[ParamType](dropout)
  val linear2 = nn.Linear[ParamType](d_ff, d_model)
  
  val layer_norm2 = nn.LayerNorm[ParamType](d_model)
  val dropout2 = nn.Dropout[ParamType](dropout)
  
  def apply(mask: Int, query: Tensor[ParamType], key: Tensor[ParamType], values: Tensor[ParamType], apply_pos: Boolean = true): Tensor[ParamType] = {
    forward(mask, query, key, values, apply_pos)
  }
  
  def forward(mask: Int, query: Tensor[ParamType], key: Tensor[ParamType], values: Tensor[ParamType], apply_pos: Boolean = true): Tensor[ParamType] = {
    val seqlen = query.size(1)
    val batch_size = query.size(0)
    val device = query.device
    
    // Create nopeek_mask
    val nopeek_mask = torch.triu(torch.ones(Seq(1, 1, seqlen, seqlen), device = device), k = mask).long()
    val src_mask = (nopeek_mask == 0)
    
    val query2 = if (mask == 0) {
      masked_attn_head(query, key, values, mask = src_mask, zero_pad = true)
    } else {
      masked_attn_head(query, key, values, mask = src_mask, zero_pad = false)
    }
    
    var updated_query = query + dropout1(query2)
    updated_query = layer_norm1(updated_query)
    
    if (apply_pos) {
      val query3 = linear2(dropout(activation(linear1(updated_query))))
      updated_query = updated_query + dropout2(query3)
      updated_query = layer_norm2(updated_query)
    }
    
    updated_query
  }
  
  override def parameters: Seq[Parameter[ParamType]] = {
    val params = ListBuffer[Parameter[ParamType]]()
    params ++= masked_attn_head.parameters
    params ++= layer_norm1.parameters
    params ++= linear1.parameters
    params ++= linear2.parameters
    params ++= layer_norm2.parameters
    params.toSeq
  }
}

class MultiHeadAttention2[ParamType <: FloatNN: Default](
    d_model: Int,
    d_feature: Int,
    n_heads: Int,
    dropout: Float,
    kq_same: Boolean,
    seq_len: Int,
    r: Float,
    gamma: Float,
    num_buckets: Int,
    max_distance: Int,
    bias: Boolean = true
) extends HasParams[ParamType] with TensorModule[ParamType] {
  
  val d_k = d_feature
  val h = n_heads
  
  val v_linear = nn.Linear[ParamType](d_model, d_model, bias = bias)
  val k_linear = nn.Linear[ParamType](d_model, d_model, bias = bias)
  val q_linear = if (!kq_same) {
    Some(nn.Linear[ParamType](d_model, d_model, bias = bias))
  } else None
  val dropout_layer = nn.Dropout[ParamType](dropout)
  val out_proj = nn.Linear[ParamType](d_model, d_model, bias = bias)
  
  var rel_pos_bias: Option[Tensor[ParamType]] = None
  var rotary_pe: Option[Tensor[ParamType]] = None
  
  // Initialize alibi
  val maxpos = 1000
  val attn_heads = n_heads
  
  val context_position = torch.arange(maxpos, device = torch.getDevice).reshape(maxpos, 1)
  val memory_position = torch.arange(maxpos, device = torch.getDevice).reshape(1, maxpos)
  val relative_position = (memory_position - context_position).abs().unsqueeze(0).expand(Seq(attn_heads, -1, -1))
  
  // Calculate slopes
  def get_slopes(n: Int): Array[Float] = {
    def get_slopes_power_of_2(n: Int): Array[Float] = {
      val start = math.pow(2, -math.pow(2, -(math.log(n) / math.log(2) - 3))).toFloat
      val ratio = start
      Array.tabulate(n)(i => start * math.pow(ratio, i).toFloat)
    }
    
    if (math.log(n) / math.log(2) % 1 == 0) {
      get_slopes_power_of_2(n)
    } else {
      val closest_power_of_2 = math.pow(2, math.floor(math.log(n) / math.log(2))).int()
      val slopes_power_of_2 = get_slopes_power_of_2(closest_power_of_2)
      val slopes_remaining = get_slopes(2 * closest_power_of_2).slice(0, 2 * closest_power_of_2, 2).take(n - closest_power_of_2)
      slopes_power_of_2 ++ slopes_remaining
    }
  }
  
  val slopes = torch.tensor(get_slopes(attn_heads)) * (-1.0f)
  val alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position
  val alibi_final = alibi.reshape(Seq(1, attn_heads, maxpos, maxpos))
  
  // Reset parameters
  _reset_parameters()
  
  private def _reset_parameters(): Unit = {
    torch.nn.init.xavier_uniform_(k_linear.weight())
    torch.nn.init.xavier_uniform_(v_linear.weight())
    q_linear.foreach(torch.nn.init.xavier_uniform_(_.weight()))
    
    if (bias) {
      torch.nn.init.constant_(k_linear.bias(), 0.0f)
      torch.nn.init.constant_(v_linear.bias(), 0.0f)
      q_linear.foreach(torch.nn.init.constant_(_.bias(), 0.0f))
      torch.nn.init.constant_(out_proj.bias(), 0.0f)
    }
  }
  
  def apply(query: Tensor[ParamType], key: Tensor[ParamType], values: Tensor[ParamType], mask: Tensor[ParamType], zero_pad: Boolean): Tensor[ParamType] = {
    forward(query, key, values, mask, zero_pad)
  }
  
  def forward(query: Tensor[ParamType], key: Tensor[ParamType], values: Tensor[ParamType], mask: Tensor[ParamType], zero_pad: Boolean): Tensor[ParamType] = {
    val bs = query.size(0)
    val half_h = h / 2
    
    val k_proj = k_linear(key).reshape(Seq(bs, -1, h, d_k))
    val q_proj = if (!kq_same) {
      q_linear.get(query).reshape(Seq(bs, -1, h, d_k))
    } else {
      k_linear(query).reshape(Seq(bs, -1, h, d_k))
    }
    val v_proj = v_linear(values).reshape(Seq(bs, -1, h, d_k))
    
    // Transpose to get dimensions bs * h * sl * d_model
    val k_transposed = k_proj.transpose(1, 2)
    val q_transposed = q_proj.transpose(1, 2)
    val v_transposed = v_proj.transpose(1, 2)
    
    // Split heads
    val q_half1 = q_transposed.slice(1, 0, half_h)
    val k_half1 = k_transposed.slice(1, 0, half_h)
    val v_half1 = v_transposed.slice(1, 0, half_h)
    
    val q_half2 = q_transposed.slice(1, half_h, h)
    val k_half2 = k_transposed.slice(1, half_h, h)
    val v_half2 = v_transposed.slice(1, half_h, h)
    
    val alibi_half1 = alibi_final.slice(1, 0, half_h)
    val alibi_half2 = alibi_final.slice(1, half_h, h)
    
    // Compute attention scores
    val scores = attention(q_half1, k_half1, v_half1, d_k, mask, dropout_layer, zero_pad, alibi_half1, rel_pos_bias, rotary_pe)
    val scores_hakt = attention_hakt(q_half2, k_half2, v_half2, d_k, mask, dropout_layer, zero_pad, alibi_half2, r, gamma, rel_pos_bias, rotary_pe)
    
    // Concatenate scores
    val scores_concat = torch.cat(Seq(scores, scores_hakt), dim = 1)
    
    // Concatenate heads and put through final linear layer
    val concat = scores_concat.transpose(1, 2).reshape(Seq(bs, -1, d_model))
    val output = out_proj(concat)
    
    output
  }
  
  override def parameters: Seq[Parameter[ParamType]] = {
    val params = ListBuffer[Parameter[ParamType]]()
    params ++= v_linear.parameters
    params ++= k_linear.parameters
    q_linear.foreach(params ++= _.parameters)
    params ++= out_proj.parameters
    params.toSeq
  }
}

object attention {
  def apply[ParamType <: FloatNN: Default](
    q: Tensor[ParamType],
    k: Tensor[ParamType],
    v: Tensor[ParamType],
    d_k: Int,
    mask: Tensor[ParamType],
    dropout: Dropout[ParamType],
    zero_pad: Boolean,
    alibi: Tensor[ParamType],
    rel_pos_bias: Option[Tensor[ParamType]],
    rotary_pe: Option[Tensor[ParamType]]
  ): Tensor[ParamType] = {
    // Calculate attention scores
    val scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k).toFloat
    
    val seq_len = scores.size(-1)
    val device = q.device
    
    // Add alibi
    val alibi_slice = alibi.slice(2, 0, seq_len).slice(3, 0, seq_len)
    val scores_with_alibi = scores + alibi_slice
    
    val bs = scores_with_alibi.size(0)
    val head = scores_with_alibi.size(1)
    
    // Apply mask
    val masked_scores = scores_with_alibi.masked_fill(mask == 0.0f, -1e32f)
    
    // Apply softmax
    val softmax_scores = F.softmax(masked_scores, dim = -1)
    
    // Apply zero padding if needed
    val padded_scores = if (zero_pad) {
      val pad_zero = torch.zeros(Seq(bs, head, 1, seq_len), device = device)
      torch.cat(Seq(pad_zero, softmax_scores.slice(2, 1, softmax_scores.size(2))), dim = 2)
    } else {
      softmax_scores
    }
    
    // Apply dropout
    val dropped_scores = dropout(padded_scores)
    
    // Calculate output
    torch.matmul(dropped_scores, v)
  }
}

object attention_hakt {
  def apply[ParamType <: FloatNN: Default](
    q: Tensor[ParamType],
    k: Tensor[ParamType],
    v: Tensor[ParamType],
    d_k: Int,
    mask: Tensor[ParamType],
    dropout: Dropout[ParamType],
    zero_pad: Boolean,
    alibi: Tensor[ParamType],
    r: Float,
    gamma: Float,
    rel_pos_bias: Option[Tensor[ParamType]],
    rotary_pe: Option[Tensor[ParamType]]
  ): Tensor[ParamType] = {
    // Calculate penumbral attention scores
    val scores = penumbral(q, k, r, gamma) / math.sqrt(d_k).toFloat
    
    val seq_len = scores.size(-1)
    val device = q.device
    
    // Add alibi
    val alibi_slice = alibi.slice(2, 0, seq_len).slice(3, 0, seq_len)
    val scores_with_alibi = scores + alibi_slice
    
    val bs = scores_with_alibi.size(0)
    val head = scores_with_alibi.size(1)
    
    // Apply mask
    val masked_scores = scores_with_alibi.masked_fill(mask == 0.0f, -1e32f)
    
    // Apply softmax
    val softmax_scores = F.softmax(masked_scores, dim = -1)
    
    // Apply zero padding if needed
    val padded_scores = if (zero_pad) {
      val pad_zero = torch.zeros(Seq(bs, head, 1, seq_len), device = device)
      torch.cat(Seq(pad_zero, softmax_scores.slice(2, 1, softmax_scores.size(2))), dim = 2)
    } else {
      softmax_scores
    }
    
    // Apply dropout
    val dropped_scores = dropout(padded_scores)
    
    // Calculate output
    torch.matmul(dropped_scores, v)
  }
}

object map_psi {
  def apply[ParamType <: FloatNN: Default](x: Tensor[ParamType], r: Float): (Tensor[ParamType], Tensor[ParamType]) = {
    val x_x = x.slice(-1, 0, x.size(-1) - 1)
    val x_y = F.sigmoid(x.slice(-1, x.size(-1) - 1, x.size(-1))).squeeze(-1)
    (x_x * x_y.unsqueeze(-1) * r, x_y * r)
  }
}

object penumbral {
  def apply[ParamType <: FloatNN: Default](q: Tensor[ParamType], k: Tensor[ParamType], r: Float = 1.0f, gamma: Float = 0.1f, eps: Float = 1e-6f): Tensor[ParamType] = {
    val (q_x, q_y) = map_psi(q, r)
    val (k_x, k_y) = map_psi(k, r)
    
    val q_y_unsqueeze = q_y.unsqueeze(3)
    val k_y_unsqueeze = k_y.unsqueeze(2)
    
    val x_q_y = torch.sqrt((r * r - q_y_unsqueeze * q_y_unsqueeze + eps).abs())
    val x_k_y = torch.sqrt((r * r - k_y_unsqueeze * k_y_unsqueeze + eps).abs())
    
    // Calculate pairwise distance
    val pairwise_dist = torch.cdist(q_x, k_x)
    
    // Calculate lca_height
    val term1 = r * r - torch.pow((x_q_y + x_k_y - pairwise_dist) / 2.0f, 2)
    val lca_height = torch.maximum(torch.maximum(q_y_unsqueeze * q_y_unsqueeze, k_y_unsqueeze * k_y_unsqueeze), term1)
    
    // Calculate lca_height_outcone
    val numerator = pairwise_dist * pairwise_dist + k_y_unsqueeze * k_y_unsqueeze - q_y_unsqueeze * q_y_unsqueeze
    val denominator = 2.0f * pairwise_dist + eps
    val lca_height_outcone = torch.pow(numerator / denominator, 2) + q_y_unsqueeze * q_y_unsqueeze
    
    // Calculate exists_cone
    val cond1 = pairwise_dist <= x_q_y
    val cond2 = torch.pow(pairwise_dist - x_q_y, 2) + k_y_unsqueeze * k_y_unsqueeze <= r * r
    val exists_cone = torch.logical_or(cond1, cond2)
    
    // Return result
    -gamma * torch.where(exists_cone, lca_height, lca_height_outcone)
  }
}

class CosinePositionalEmbedding2[ParamType <: FloatNN: Default](d_model: Int, max_len: Int = 512) extends HasParams[ParamType] with TensorModule[ParamType] {
  val max_len_actual = 1000
  
  // Initialize positional encodings
  val pe_tensor = 0.1f * torch.randn(Seq(max_len_actual, d_model))
  val position = torch.arange(0, max_len_actual).reshape(max_len_actual, 1)
  val div_term = torch.exp(
    torch.arange(0, d_model, 2) * (-(math.log(10000.0) / d_model).toFloat)
  )
  
  // Set even and odd positions
  pe_tensor.slice(1, 0, d_model, 2) := torch.sin(position * div_term)
  pe_tensor.slice(1, 1, d_model, 2) := torch.cos(position * div_term)
  
  val pe = nn.Parameter[ParamType](pe_tensor.unsqueeze(0), requires_grad = false)
  
  def apply(x: Tensor[ParamType]): Tensor[ParamType] = {
    forward(x)
  }
  
  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    pe.slice(1, 0, x.size(1))
  }
  
  override def parameters: Seq[Parameter[ParamType]] = Seq(pe)
}

class LearnablePositionalEmbedding2[ParamType <: FloatNN: Default](d_model: Int, max_len: Int = 512) extends HasParams[ParamType] with TensorModule[ParamType] {
  val max_len_actual = 1000
  
  // Initialize learnable positional encodings
  val pe = nn.Parameter[ParamType](
    0.1f * torch.randn(Seq(1, max_len_actual, d_model)),
    requires_grad = true
  )
  
  def apply(x: Tensor[ParamType]): Tensor[ParamType] = {
    forward(x)
  }
  
  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    pe.slice(1, 0, x.size(1))
  }
  
  override def parameters: Seq[Parameter[ParamType]] = Seq(pe)
}

// Factory methods for stableKT
object stableKT {
  def apply[ParamType <: FloatNN: Default](
    mask_response: Boolean,
    pred_last: Boolean,
    mask_future: Boolean,
    length: Int,
    trans: Boolean,
    num_skills: Int,
    num_questions: Int,
    embedding_size: Int,
    num_blocks: Int,
    dropout: Float,
    d_ff: Int = 256,
    seq_len: Int = 200,
    kq_same: Int = 1,
    final_fc_dim: Int = 512,
    final_fc_dim2: Int = 256,
    num_attn_heads: Int = 8,
    separate_qr: Boolean = false,
    l2: Float = 1e-5f,
    r: Float = 1.0f,
    gamma: Float = 1.0f,
    num_buckets: Int = 32,
    max_distance: Int = 100
  ): stableKT[ParamType] = {
    new stableKT(
      mask_response, pred_last, mask_future, length, trans,
      num_skills, num_questions, embedding_size, num_blocks,
      dropout, d_ff, seq_len, kq_same, final_fc_dim,
      final_fc_dim2, num_attn_heads, separate_qr, l2,
      r, gamma, num_buckets, max_distance
    )
  }
}
