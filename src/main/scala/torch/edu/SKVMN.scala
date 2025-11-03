package torch.edu

import torch.*
import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn as nn
import torch.nn.*
import scala.collection.mutable.ListBuffer

class DKVMNHeadGroup[ParamType <: FloatNN: Default](
    memory_size: Int,
    memory_state_dim: Int,
    is_write: Boolean
) extends HasParams[ParamType] {
  
  val erase = if (is_write) {
    Some(Linear[ParamType](memory_state_dim, memory_state_dim, bias = true))
  } else None
  
  val add = if (is_write) {
    Some(Linear[ParamType](memory_state_dim, memory_state_dim, bias = true))
  } else None
  
  // 初始化参数
  if (is_write) {
    nn.init.kaiming_normal_(erase.get.weight)
    nn.init.kaiming_normal_(add.get.weight)
    nn.init.constant_(erase.get.bias, 0f)
    nn.init.constant_(add.get.bias, 0f)
  }
  
  override def params: Seq[Tensor[ParamType]] = {
    (erase.map(_.params).getOrElse(Nil) ++ add.map(_.params).getOrElse(Nil)).toSeq
  }
  
  def addressing(control_input: Tensor[ParamType], memory: Tensor[ParamType]): Tensor[ParamType] = {
    // control_input: (batch_size, control_state_dim)
    // memory: (memory_size, memory_state_dim)
    val similarity_score = torch.matmul(control_input, memory.t)
    F.softmax(similarity_score, dim = 1)
  }
  
  def read(
      memory: Tensor[ParamType], 
      control_input: Option[Tensor[ParamType]] = None, 
      read_weight: Option[Tensor[ParamType]] = None
  ): Tensor[ParamType] = {
    // memory: (batch_size, memory_size, memory_state_dim)
    // control_input: (batch_size, control_state_dim)
    // read_weight: (batch_size, memory_size)
    
    val rw = read_weight.getOrElse {
      addressing(control_input.get, memory)
    }
    
    val read_weight_reshaped = rw.view(-1, 1)
    val memory_reshaped = memory.view(-1, memory_state_dim)
    val rc = torch.mul(read_weight_reshaped, memory_reshaped)
    val read_content = rc.view(-1, memory_size, memory_state_dim)
    torch.sum(read_content, dim = 1)
  }
  
  def write(
      control_input: Tensor[ParamType], 
      memory: Tensor[ParamType], 
      write_weight: Option[Tensor[ParamType]] = None
  ): Tensor[ParamType] = {
    // control_input: (batch_size, control_state_dim)
    // write_weight: (batch_size, memory_size)
    // memory: (batch_size, memory_size, memory_state_dim)
    
    require(is_write, "Cannot write with a read-only head")
    
    val ww = write_weight.getOrElse {
      addressing(control_input, memory)
    }
    
    val erase_signal = torch.sigmoid(erase.get(control_input))
    val add_signal = torch.tanh(add.get(control_input))
    
    val erase_reshape = erase_signal.view(-1, 1, memory_state_dim)
    val add_reshape = add_signal.view(-1, 1, memory_state_dim)
    val write_weight_reshape = ww.view(-1, memory_size, 1)
    
    val erase_mul = torch.mul(erase_reshape, write_weight_reshape)
    val add_mul = torch.mul(add_reshape, write_weight_reshape)
    
    // 确保内存和操作在同一设备上
    val memory_device = memory.device
    val new_memory = if (add_mul.shape(0) < memory.shape(0)) {
      val sub_memory = memory.slice(0, 0, add_mul.shape(0), memory.shape(0), 1)
      torch.cat(
        Seq(
          torch.mul(sub_memory, (1f - erase_mul)) + add_mul,
          memory.slice(0, add_mul.shape(0), memory.shape(0), memory.shape(0), 1)
        ), 
        dim = 0
      )
    } else {
      torch.mul(memory, (1f - erase_mul)) + add_mul
    }
    
    new_memory.to(memory_device)
  }
}

class DKVMN2[ParamType <: FloatNN: Default](
    memory_size: Int,
    memory_key_state_dim: Int,
    memory_value_state_dim: Int,
    init_memory_key: Tensor[ParamType]
) extends HasParams[ParamType] {
  
  val key_head = DKVMNHeadGroup[ParamType](memory_size, memory_key_state_dim, is_write = false)
  val value_head = DKVMNHeadGroup[ParamType](memory_size, memory_value_state_dim, is_write = true)
  
  // 使用可变参数以便在训练过程中更新
  var memory_key: Tensor[ParamType] = init_memory_key
  
  override def params: Seq[Tensor[ParamType]] = {
    key_head.params ++ value_head.params
  }
  
  def attention(control_input: Tensor[ParamType]): Tensor[ParamType] = {
    key_head.addressing(control_input, memory_key)
  }
  
  def read(read_weight: Tensor[ParamType], memory_value: Tensor[ParamType]): Tensor[ParamType] = {
    value_head.read(memory_value, read_weight = Some(read_weight))
  }
  
  def write(
      write_weight: Tensor[ParamType], 
      control_input: Tensor[ParamType], 
      memory_value: Tensor[ParamType]
  ): Tensor[ParamType] = {
    value_head.write(control_input, memory_value, write_weight = Some(write_weight))
  }
}

class SKVMN[ParamType <: FloatNN: Default](
    pred_last: Boolean,
    mask_future: Boolean,
    length: Int,
    trans: Boolean,
    num_skills: Int,
    dim_s: Int,
    size_m: Int,
    dropout: Float = 0.2f,
    use_onehot: Boolean = false
) extends TensorModule[ParamType] {
  
  val k_emb_layer = Embedding[ParamType](num_skills, dim_s)
  val x_emb_layer = Embedding[ParamType](2 * num_skills + 1, dim_s)
  
  // 初始化记忆参数
  val Mk = Tensor[ParamType](size_m, dim_s)
  val Mv0 = Tensor[ParamType](size_m, dim_s)
  nn.init.kaiming_normal_(Mk)
  nn.init.kaiming_normal_(Mv0)
  
  val mem = DKVMN[ParamType](size_m, dim_s, dim_s, Mk)
  
  val a_embed = if (use_onehot) {
    Linear[ParamType](num_skills + dim_s, dim_s, bias = true)
  } else {
    Linear[ParamType](dim_s * 2, dim_s, bias = true)
  }
  
  val v_emb_layer = Embedding[ParamType](dim_s * 2, dim_s)
  val f_layer = Linear[ParamType](dim_s * 2, dim_s)
  
  // LSTM参数
  val hx = Tensor[ParamType](1, dim_s)
  val cx = Tensor[ParamType](1, dim_s)
  nn.init.kaiming_normal_(hx)
  nn.init.kaiming_normal_(cx)
  
  val dropout_layer = Dropout[ParamType](dropout)
  val p_layer = if (trans) {
    Linear[ParamType](dim_s, num_skills)
  } else {
    Linear[ParamType](dim_s, 1)
  }
  
  val lstm_cell = LSTMCell[ParamType](dim_s, dim_s)
  val loss_fn = BCELoss(reduction = "mean")
  
  // 添加临时变量来存储seqlen
  var seqlen: Int = 0
  
  override def params: Seq[Tensor[ParamType]] = {
    Seq(
      k_emb_layer, x_emb_layer, Mk, Mv0, a_embed, v_emb_layer, 
      f_layer, hx, cx, p_layer, lstm_cell
    ).flatMap(_.params) ++ dropout_layer.params ++ mem.params ++ loss_fn.params
  }
  
  def ut_mask(seq_len: Int): Tensor[ParamType] = {
    torch.triu(torch.ones[ParamType](seq_len, seq_len), diagonal = 0).to(dtype = torch.bool)
  }
  
  def triangular_layer(
      correlation_weight: Tensor[ParamType], 
      batch_size: Int = 64,
      a: Float = 0.075f, 
      b: Float = 0.088f, 
      c: Float = 1.00f
  ): Tensor[ParamType] = {
    val device = correlation_weight.device
    
    // 重塑和处理相关权重
    val cw_reshaped = correlation_weight.view(batch_size * seqlen, -1)
    val cw_flattened = torch.cat(
      (0 until cw_reshaped.shape(0)).map(i => cw_reshaped.slice(0, i, i + 1)), 
      dim = 1
    ).unsqueeze(0)
    
    // 计算w'
    val part1 = (cw_flattened - a) / (b - a)
    val part2 = (c - cw_flattened) / (c - b)
    val cw_combined = torch.cat(Seq(part1, part2), dim = 0)
    val cw_min, _ = torch.min(cw_combined, 0)
    
    val w0 = torch.zeros[ParamType](cw_min.shape(0)).to(device)
    val cw_with_zero = torch.cat(Seq(cw_min.unsqueeze(0), w0.unsqueeze(0)), dim = 0)
    val cw_max, _ = torch.max(cw_with_zero, 0)
    
    // 构建身份向量
    var identity_vector_batch = torch.zeros[ParamType](cw_max.shape(0)).to(device)
    identity_vector_batch = identity_vector_batch.masked_fill(cw_max.lt(0.1f), 0f)
    identity_vector_batch = identity_vector_batch.masked_fill(cw_max.ge(0.1f), 1f)
    val _identity_vector_batch = identity_vector_batch.masked_fill(cw_max.ge(0.6f), 2f)
    
    // 重塑为批处理格式
    identity_vector_batch = _identity_vector_batch.view(batch_size * seqlen, -1)
    identity_vector_batch = identity_vector_batch.reshape(batch_size, seqlen, -1)
    
    // 计算距离矩阵
    val iv_square_norm = torch.sum(torch.pow(identity_vector_batch, 2f), dim = 2, keepdim = true)
    val iv_square_norm_repeated = iv_square_norm.repeat(Seq(1, 1, iv_square_norm.shape(1)))
    
    val unique_iv_square_norm = torch.sum(torch.pow(identity_vector_batch, 2f), dim = 2, keepdim = true)
    val unique_iv_square_norm_repeated = unique_iv_square_norm.repeat(Seq(1, 1, seqlen)).transpose(2, 1)
    
    val iv_matrix_product = torch.bmm(identity_vector_batch, identity_vector_batch.transpose(2, 1))
    var iv_distances = iv_square_norm_repeated + unique_iv_square_norm_repeated - 2f * iv_matrix_product
    
    // 应用掩码
    iv_distances = torch.where(iv_distances.gt(0f), torch.tensor(-1e32f, device = device), iv_distances)
    val masks = ut_mask(iv_distances.shape(1)).to(device)
    val mask_iv_distances = iv_distances.masked_fill(masks, value = torch.tensor(-1e32f, device = device))
    
    // 添加索引矩阵
    val idx_matrix = torch.arange(0, seqlen * seqlen, 1)
      .reshape(seqlen, -1)
      .repeat(batch_size, 1, 1)
      .to(device)
    val final_iv_distance = mask_iv_distances + idx_matrix
    
    // 获取topk值
    val (values, indices) = torch.topk(final_iv_distance, 1, dim = 2, largest = true)
    
    // 处理结果以获取身份索引
    val _values = values.permute(1, 0, 2)
    val _indices = indices.permute(1, 0, 2)
    val batch_identity_indices = (_values >= 0f).nonzero()
    
    val identity_idx = ListBuffer[Tensor[ParamType]]()
    for (identity_indices <- batch_identity_indices) {
      val pre_idx = _indices.slice(identity_indices(0), identity_indices(0) + 1)
        .slice(identity_indices(1), identity_indices(1) + 1)
      val idx = torch.cat(Seq(identity_indices.slice(0, -1), pre_idx.squeeze()), dim = -1)
      identity_idx.append(idx)
    }
    
    if (identity_idx.nonEmpty) {
      torch.stack(identity_idx, dim = 0)
    } else {
      torch.tensor[ParamType](Seq.empty).to(device)
    }
  }
  
  override def forward(feed_dict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = {
    val q = feed_dict("skills")
    val r = feed_dict("responses")
    val masked_r = r * (r.gt(-1f)).long()
    
    var q_input: Tensor[ParamType] = null
    var r_input: Tensor[ParamType] = null
    var cshft: Tensor[ParamType] = null
    
    if (trans) {
      cshft = q.slice(1, length, q.shape(1), q.shape(1), 1)
      q_input = q.slice(1, 0, q.shape(1) - length, q.shape(1), 1)
      r_input = masked_r.slice(1, 0, masked_r.shape(1) - length, masked_r.shape(1), 1)
    } else if (mask_future) {
      val attention_mask = feed_dict("attention_mask")
      // 在Scala中修改张量的一部分
      val mask = attention_mask.clone()
      mask.slice(1, mask.shape(1) - length, mask.shape(1), mask.shape(1), 1).fill_(0f)
      q_input = q * mask
      r_input = r * mask
    } else if (pred_last) {
      q_input = q
      r_input = masked_r
    } else {
      q_input = q
      r_input = r
    }
    
    val bs = q_input.shape(0)
    seqlen = q_input.shape(1)
    val device = q_input.device
    
    // 计算x和嵌入
    val x = q_input + num_skills * r_input
    val k = k_emb_layer(q_input)
    
    // 处理one-hot情况
    var r_onehot_content: Option[Tensor[ParamType]] = None
    if (use_onehot) {
      val q_data = q.reshape(bs * seqlen, 1)
      val r_onehot = torch.zeros[ParamType](bs * seqlen, num_skills).long().to(device)
      val r_data = masked_r.unsqueeze(2).expand(Seq(-1, -1, num_skills))
        .reshape(bs * seqlen, num_skills)
      r_onehot_content = Some(r_onehot.scatter(1, q_data, r_data).reshape(bs, seqlen, -1))
    }
    
    // 初始化记忆和存储列表
    val value_read_content_l = ListBuffer[Tensor[ParamType]]()
    val input_embed_l = ListBuffer[Tensor[ParamType]]()
    val correlation_weight_list = ListBuffer[Tensor[ParamType]]()
    val ft_list = ListBuffer[Tensor[ParamType]]()
    
    // 初始化记忆值
    var mem_value = Mv0.unsqueeze(0).repeat(Seq(bs, 1, 1)).to(device)
    
    // 处理每个时间步
    for (i <- 0 until seqlen) {
      // 注意力计算
      val q_t = k.permute(1, 0, 2).slice(0, i, i + 1).squeeze(0)
      val correlation_weight = mem.attention(q_t).to(device)
      
      // 读取过程
      val read_content = mem.read(correlation_weight, mem_value)
      
      // 保存中间数据
      correlation_weight_list.append(correlation_weight)
      value_read_content_l.append(read_content)
      input_embed_l.append(q_t)
      
      // 计算f
      val batch_predict_input = torch.cat(Seq(read_content, q_t), dim = 1)
      val f = torch.tanh(f_layer(batch_predict_input))
      ft_list.append(f)
      
      // 准备写入内容
      val y = if (use_onehot) {
        r_onehot_content.get.slice(1, i, i + 1).squeeze(1)
      } else {
        x_emb_layer(x.slice(1, i, i + 1).squeeze(1))
      }
      
      // 写入记忆
      val write_embed = torch.cat(Seq(f, y), dim = 1)
      val write_embed_processed = a_embed(write_embed).to(device)
      mem_value = mem.write(correlation_weight, write_embed_processed, mem_value)
    }
    
    // 处理权重和特征
    val w = torch.cat(
      correlation_weight_list.map(cw => cw.unsqueeze(1)), 
      dim = 1
    )
    val ft = torch.stack(ft_list, dim = 0)
    
    // 获取顺序依赖关系
    val idx_values = triangular_layer(w, bs)
    
    // Hop-LSTM处理
    val hidden_state = ListBuffer[Tensor[ParamType]]()
    val cell_state = ListBuffer[Tensor[ParamType]]()
    var hx_current = hx.repeat(Seq(bs, 1))
    var cx_current = cx.repeat(Seq(bs, 1))
    
    var remaining_idx_values = idx_values
    
    for (i <- 0 until seqlen) {
      // 检查是否需要替换隐藏状态
      for (j <- 0 until bs) {
        if (remaining_idx_values.shape(0) > 0 && i == remaining_idx_values.slice(0, 0, 1)(0)(0).long() &&
            j == remaining_idx_values.slice(0, 0, 1)(0)(1).long()) {
          val prev_idx = remaining_idx_values.slice(0, 0, 1)(0)(2).long()
          hx_current.slice(0, j, j + 1).copy_(hidden_state(prev_idx.int()).slice(0, j, j + 1))
          cx_current = cx_current.clone()
          cx_current.slice(0, j, j + 1).copy_(cell_state(prev_idx.int()).slice(0, j, j + 1))
          
          // 更新剩余索引
          remaining_idx_values = remaining_idx_values.slice(0, 1, remaining_idx_values.shape(0))
        }
      }
      
      // LSTM单元前向传播
      val (new_hx, new_cx) = lstm_cell(ft.slice(0, i, i + 1).squeeze(0), (hx_current, cx_current))
      hidden_state.append(new_hx)
      cell_state.append(new_cx)
      hx_current = new_hx
      cx_current = new_cx
    }
    
    // 堆叠并调整维度
    val hidden_state_stacked = torch.stack(hidden_state, dim = 0).permute(1, 0, 2)
    
    // 生成预测
    var p: Tensor[ParamType] = null
    var true_target: Tensor[ParamType] = null
    
    if (trans) {
      p = p_layer(dropout_layer(hidden_state_stacked))
      p = torch.sigmoid(p)
      // 应用one-hot掩码
      val one_hot_mask = F.one_hot(cshft.long(), num_skills)
      p = torch.sum(p * one_hot_mask, dim = -1)
      true_target = r.slice(1, length, r.shape(1)).to(dtype = torch.float32)
    } else if (mask_future || pred_last) {
      p = p_layer(dropout_layer(hidden_state_stacked))
      p = torch.sigmoid(p)
      p = p.squeeze(-1)
      p = p.slice(1, p.shape(1) - length, p.shape(1))
      true_target = r.slice(1, r.shape(1) - length, r.shape(1)).to(dtype = torch.float32)
    } else {
      p = p_layer(dropout_layer(hidden_state_stacked))
      p = torch.sigmoid(p)
      p = p.squeeze(-1)
      p = p.slice(1, length, p.shape(1))
      true_target = r.slice(1, length, r.shape(1)).to(dtype = torch.float32)
    }
    
    // 返回结果字典
    Map(
      "pred" -> p,
      "true" -> true_target
    )
  }
  
  // 应用方法调用forward
  def apply(feed_dict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = {
    forward(feed_dict)
  }
  
  // 损失计算方法
  def loss(feed_dict: Map[String, Tensor[ParamType]], out_dict: Map[String, Tensor[ParamType]]): 
      (Tensor[ParamType], Long, Double) = {
    val pred = out_dict("pred").flatten()
    val true_target = out_dict("true").flatten()
    val mask = true_target.gt(-1f)
    
    val loss_value = loss_fn(pred.masked_select(mask), true_target.masked_select(mask))
    val num_elements = pred.masked_select(mask).numel()
    val sum_true = true_target.masked_select(mask).sum().item().double()
    
    (loss_value, num_elements, sum_true)
  }
}

// 伴生对象提供工厂方法
object SKVMN {
  def apply[ParamType <: FloatNN: Default](
      pred_last: Boolean,
      mask_future: Boolean,
      length: Int,
      trans: Boolean,
      num_skills: Int,
      dim_s: Int,
      size_m: Int,
      dropout: Float = 0.2f,
      use_onehot: Boolean = false
  ): SKVMN[ParamType] = {
    new SKVMN[ParamType](pred_last, mask_future, length, trans, num_skills, dim_s, size_m, dropout, use_onehot)
  }
}
