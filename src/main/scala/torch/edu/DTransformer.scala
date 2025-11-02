package torch.edu

import torch.*
import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn as nn
import torch.nn.*
import scala.collection.mutable.ListBuffer
import scala.util.Random

val MIN_SEQ_LEN = 5

class DTransformer[ParamType <: FloatNN: Default](
    maskResponse: Boolean,
    predLast: Boolean,
    maskFuture: Boolean,
    length: Int,
    trans: Boolean,
    numSkills: Int,
    numQuestions: Int,
    embeddingSize: Int = 64,
    dFF: Int = 256,
    numAttnHeads: Int = 8,
    nKnow: Int = 16,
    numBlocks: Int = 3,
    dropout: Float = 0.3f,
    lambdaCl: Float = 0.1f,
    proj: Boolean = false,
    hardNeg: Boolean = false,
    window: Int = 1,
    shortcut: Boolean = false,
    separateQr: Boolean = false
) extends TensorModule[ParamType] with HasParams[ParamType] {
  
  val dropoutRate = dropout
  val lambdaClValue = lambdaCl
  val hardNegative = hardNeg
  val shortcutEnabled = shortcut
  val nLayers = numBlocks
  val windowSize = window
  val embedL = embeddingSize
  
  // Embedding layers
  val qDiffEmbed = if (numQuestions > 0) Some(nn.Embedding(numSkills + 1, embeddingSize)) else None
  val sDiffEmbed = if (numQuestions > 0) Some(nn.Embedding(2, embeddingSize)) else None
  val pDiffEmbed = if (numQuestions > 0) Some(nn.Embedding(numQuestions + 1, 1)) else None
  
  val qEmbed = nn.Embedding(numSkills, embedL)
  val sEmbed = if (separateQr) {
    nn.Embedding(2 * numSkills + 1, embedL) // interaction emb
  } else {
    nn.Embedding(2, embedL)
  }
  
  // Transformer Encoder layers
  val nHeads = numAttnHeads
  val block1 = new DTransformerLayer[ParamType](embeddingSize, nHeads, dropout)
  val block2 = new DTransformerLayer[ParamType](embeddingSize, nHeads, dropout)
  val block3 = new DTransformerLayer[ParamType](embeddingSize, nHeads, dropout)
  val block4 = new DTransformerLayer[ParamType](embeddingSize, nHeads, dropout, kqSame = false)
  
  // Knowledge parameters
  val knowParams = Tensor.randn[ParamType](Seq(nKnow, embeddingSize))
  
  // Output Layer
  val out = if (trans) {
    nn.Sequential(
      nn.Linear(embeddingSize * 2, dFF),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(dFF, dFF / 2),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(dFF / 2, numSkills)
    )
  } else {
    nn.Sequential(
      nn.Linear(embeddingSize * 2, dFF),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(dFF, dFF / 2),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(dFF / 2, 1)
    )
  }
  
  // CL Linear Layer
  val projLayer = if (proj) {
    Some(nn.Sequential(
      nn.Linear(embeddingSize, embeddingSize),
      nn.GELU()
    ))
  } else {
    None
  }
  
  val lossFn = nn.BCELoss(reduction = "mean")
  
  // Initialize
  reset()
  
  // Collect all parameters
  override val params: Seq[Tensor[ParamType]] = {
    val paramsList = ListBuffer[Tensor[ParamType]]()
    
    qDiffEmbed.foreach(e => paramsList.appendAll(e.params))
    sDiffEmbed.foreach(e => paramsList.appendAll(e.params))
    pDiffEmbed.foreach(e => paramsList.appendAll(e.params))
    
    paramsList.appendAll(qEmbed.params)
    paramsList.appendAll(sEmbed.params)
    
    paramsList.appendAll(block1.params)
    paramsList.appendAll(block2.params)
    paramsList.appendAll(block3.params)
    paramsList.appendAll(block4.params)
    
    paramsList.append(knowParams)
    
    paramsList.appendAll(out.params)
    projLayer.foreach(l => paramsList.appendAll(l.params))
    
    paramsList.toSeq
  }
  
  // Reset parameters
  def reset(): Unit = {
    for (p <- params) {
      if (numQuestions > 0 && p.shape(0) == numQuestions + 1) {
        p.data.fill_(0.0f)
      }
    }
  }
  
  // Forward method for the main transformer processing
  def forward(qEmb: Tensor[ParamType], sEmb: Tensor[ParamType], lens: Tensor[ParamType]): 
      (Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]) = {
    if (shortcutEnabled) {
      // AKT shortcut path
      val (hq, _) = block1(qEmb, qEmb, qEmb, lens, peekCur = true)
      val (hs, scores) = block2(sEmb, sEmb, sEmb, lens, peekCur = true)
      val (result, _) = block3(hq, hq, hs, lens, peekCur = false)
      (result, scores, Tensor.empty[ParamType]())
    } else if (nLayers == 1) {
      val hq = qEmb
      val (p, qScores) = block1(qEmb, qEmb, sEmb, lens, peekCur = true)
      (p, qScores, Tensor.empty[ParamType]())
    } else if (nLayers == 2) {
      val hq = qEmb
      val (hs, _) = block1(sEmb, sEmb, sEmb, lens, peekCur = true)
      val (p, qScores) = block2(hq, hq, hs, lens, peekCur = true)
      (p, qScores, Tensor.empty[ParamType]())
    } else {
      // Default path with 3 or more layers
      val (hq, _) = block1(qEmb, qEmb, qEmb, lens, peekCur = true)
      val (hs, _) = block2(sEmb, sEmb, sEmb, lens, peekCur = true)
      val (p, qScores) = block3(hq, hq, hs, lens, peekCur = true)
      
      val bs = p.shape(0)
      val seqlen = p.shape(1)
      
      // Prepare knowledge query
      val query = knowParams.unsqueeze(0).unsqueeze(2)
        .expand(bs, -1, seqlen, -1)
        .contiguous()
        .view(bs * nKnow, seqlen, embeddingSize)
      
      val hqExpanded = hq.unsqueeze(1)
        .expand(-1, nKnow, -1, -1)
        .reshape(query.shape)
      
      val pExpanded = p.unsqueeze(1)
        .expand(-1, nKnow, -1, -1)
        .reshape(query.shape)
      
      // Apply knowledge block
      val lensRepeated = lens.repeat(nKnow)
      val (z, kScores) = block4(query, hqExpanded, pExpanded, lensRepeated, peekCur = false)
      
      // Reshape outputs
      val zReshaped = z.view(bs, nKnow, seqlen, embeddingSize)
        .transpose(1, 2)
        .contiguous()
        .view(bs, seqlen, -1)
      
      val kScoresReshaped = kScores.view(bs, nKnow, nHeads, seqlen, seqlen)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
      
      (zReshaped, qScores, kScoresReshaped)
    }
  }
  
  // Base embedding method
  def baseEmb(qData: Tensor[ParamType], target: Tensor[ParamType]): 
      (Tensor[ParamType], Tensor[ParamType]) = {
    val qEmbedData = qEmbed(qData.toLong)
    
    val qaEmbedData = if (separateQr) {
      val qaData = qData + numSkills * target
      sEmbed(qaData.toLong)
    } else {
      sEmbed(target.toLong) + qEmbedData
    }
    
    (qEmbedData, qaEmbedData)
  }
  
  // Full embedding with question difficulty if available
  def embedding(qData: Tensor[ParamType], target: Tensor[ParamType], pidData: Option[Tensor[ParamType]] = None): 
      (Tensor[ParamType], Tensor[ParamType], Tensor[ParamType], Option[Tensor[ParamType]]) = {
    val lens = (target >= 0).sum(dim = 1)
    var (qEmbedData, qaEmbedData) = baseEmb(qData, target)
    
    var pidEmbedData: Option[Tensor[ParamType]] = None
    
    if (numQuestions > 0 && pidData.isDefined) {
      val pid = pidData.get
      
      qDiffEmbed.foreach { qDiff =>
        val qEmbedDiffData = qDiff(qData.toLong)
        pDiffEmbed.foreach {
          pDiff => 
            pidEmbedData = Some(pDiff(pid.toLong))
            pidEmbedData.foreach { p =>
              qEmbedData = qEmbedData + p * qEmbedDiffData
            }
        }
      }
      
      sDiffEmbed.foreach {
        sDiff =>
          val qaEmbedDiffData = sDiff(target.toLong)
          pidEmbedData.foreach {
            p =>
              if (separateQr) {
                qaEmbedData = qaEmbedData + p * qaEmbedDiffData
              } else {
                qDiffEmbed.foreach {
                  qDiff =>
                    val qEmbedDiffData = qDiff(qData.toLong)
                    qaEmbedData = qaEmbedData + p * (qaEmbedDiffData + qEmbedDiffData)
                }
              }
          }
      }
    }
    
    (qEmbedData, qaEmbedData, lens, pidEmbedData)
  }
  
  // Readout mechanism
  def readout(z: Tensor[ParamType], query: Tensor[ParamType]): Tensor[ParamType] = {
    val bs = query.shape(0)
    val seqlen = query.shape(1)
    
    val key = knowParams.unsqueeze(0).unsqueeze(0)
      .expand(bs, seqlen, -1, -1)
      .view(bs * seqlen, nKnow, -1)
    
    val value = z.reshape(bs * seqlen, nKnow, -1)
    
    val beta = torch.matmul(
      key,
      query.reshape(bs * seqlen, -1, 1)
    ).view(bs * seqlen, 1, nKnow)
    
    val alpha = F.softmax(beta, dim = -1)
    
    torch.matmul(alpha, value).view(bs, seqlen, -1)
  }
  
  // Prediction method
  def predict(q: Tensor[ParamType], s: Tensor[ParamType], pid: Option[Tensor[ParamType]] = None, n: Int = 1): 
      (Tensor[ParamType], Tensor[ParamType], Tensor[ParamType], Tensor[ParamType], Tensor[ParamType], (Tensor[ParamType], Tensor[ParamType])) = {
    val (qEmb, sEmb, lens, pDiff) = embedding(q, s, pid)
    val (z, qScores, kScores) = this.forward(qEmb, sEmb, lens)
    
    val h = if (shortcutEnabled) {
      assert(n == 1, "AKT does not support T+N prediction")
      z
    } else {
      val query = qEmb.slice(1, n - 1, qEmb.shape(1))
      readout(z.slice(1, 0, query.shape(1)), query)
    }
    
    val concatQ = torch.cat(Seq(qEmb, h), dim = -1)
    
    val output = if (trans) {
      out(concatQ)
    } else {
      out(concatQ).squeeze(-1)
    }
    
    val regLoss = pDiff.map(p => (p.pow(2)).mean() * 1e-3f).getOrElse(torch.zeros())
    
    (output, concatQ, z, qEmb, regLoss, (qScores, kScores))
  }
  
  // Get loss for prediction
  def getLoss(device: Device, q: Tensor[ParamType], s: Tensor[ParamType], pids: Option[Tensor[ParamType]] = None, qCl: Boolean = false): 
      Tensor[ParamType] | (Tensor[ParamType], Tensor[ParamType]) = {
    val (qDevice, sDevice) = (q.to(device), s.to(device))
    val pidDevice = pids.map(_.to(device))
    
    val (output, _, _, _, regLoss, _) = predict(qDevice, sDevice, pidDevice)
    val m = nn.Sigmoid()
    val preds = m(output)
    
    if (qCl) {
      val maskedLabels = sDevice.where(sDevice >= 0, sDevice, Tensor.empty[ParamType]())
      val maskedLogits = output.where(sDevice >= 0, output, Tensor.empty[ParamType]())
      F.binarycross_entropyWithLogits(maskedLogits, maskedLabels, reduction = "mean") + regLoss
    } else {
      (preds, regLoss)
    }
  }
  
  // Get contrastive learning loss
  def getClLoss(feedDict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = {
    val pid = feedDict("questions")
    val r = feedDict("responses")
    val q = feedDict("skills")
    val attentionMask = feedDict("attention_mask")
    val s = r * r.gt(-1)
    
    var (pidProcessed, sProcessed, qProcessed, cshft) = 
      if (trans) {
        val pidSliced = pid.slice(1, 0, pid.shape(1) - length)
        val sSliced = s.slice(1, 0, s.shape(1) - length)
        val qSliced = q.slice(1, 0, q.shape(1) - length)
        val cshftTensor = q.slice(1, length, q.shape(1))
        (pidSliced, sSliced, qSliced, Some(cshftTensor))
      } else if (maskFuture) {
        val maskModified = attentionMask.clone()
        maskModified.slice(1, maskModified.shape(1) - length, maskModified.shape(1)).fill_(0.0f)
        val pidMasked = pid * maskModified
        val sMasked = s * maskModified
        val qMasked = q * maskModified
        (pidMasked, sMasked, qMasked, None)
      } else if (maskResponse) {
        val maskModified = attentionMask.clone()
        maskModified.slice(1, maskModified.shape(1) - length, maskModified.shape(1)).fill_(0.0f)
        val sMasked = s * maskModified
        (pid, sMasked, q, None)
      } else {
        (pid, s, q, None)
      }
    
    val bs = sProcessed.shape(0)
    
    // Input data preprocessing
    val pidOpt = if (pidProcessed.shape(1) == 0 || numQuestions == 0) None else Some(pidProcessed)
    
    // Augmentation
    val qAugmented = qProcessed.clone()
    val sAugmented = sProcessed.clone()
    val pidAugmentedOpt = pidOpt.map(_.clone())
    
    // Manipulate order
    for (b <- 0 until bs) {
      val len = (sProcessed.slice(0, b, b + 1) >= 0).sum().int()
      if (len > 1) {
        val idx = Random.shuffle((0 until len - 1).toList)
          .take(math.max(1, (len * dropoutRate).int()))
        
        for (i <- idx) {
          // Swap i and i+1
          val qTemp = qAugmented.slice(0, b, b+1).slice(1, i, i+1).clone()
          qAugmented.slice(0, b, b+1).slice(1, i, i+1).copy_(qAugmented.slice(0, b, b+1).slice(1, i+1, i+2))
          qAugmented.slice(0, b, b+1).slice(1, i+1, i+2).copy_(qTemp)
          
          val sTemp = sAugmented.slice(0, b, b+1).slice(1, i, i+1).clone()
          sAugmented.slice(0, b, b+1).slice(1, i, i+1).copy_(sAugmented.slice(0, b, b+1).slice(1, i+1, i+2))
          sAugmented.slice(0, b, b+1).slice(1, i+1, i+2).copy_(sTemp)
          
          pidAugmentedOpt.foreach {
            pidAugmented =>
              val pidTemp = pidAugmented.slice(0, b, b+1).slice(1, i, i+1).clone()
              pidAugmented.slice(0, b, b+1).slice(1, i, i+1).copy_(pidAugmented.slice(0, b, b+1).slice(1, i+1, i+2))
              pidAugmented.slice(0, b, b+1).slice(1, i+1, i+2).copy_(pidTemp)
          }
        }
      }
    }
    
    // Hard negative
    val sFlip = if (hardNegative) sProcessed.clone() else sAugmented.clone()
    for (b <- 0 until bs) {
      val len = (sProcessed.slice(0, b, b + 1) >= 0).sum().int()
      if (len > 0) {
        val idx = Random.shuffle((0 until len).toList)
          .take(math.max(1, (len * dropoutRate).int()))
        
        for (i <- idx) {
          sFlip.slice(0, b, b+1).slice(1, i, i+1).copy_(1.0f - sFlip.slice(0, b, b+1).slice(1, i, i+1))
        }
      }
    }
    
    if (!hardNegative) {
      sAugmented.copy_(sFlip)
    }
    
    // Model predictions
    val (logits, concatQ, z1, qEmb, regLoss, _) = predict(qProcessed, sProcessed, pidOpt)
    val (_, _, z2, _, _, _) = predict(qAugmented, sAugmented, pidAugmentedOpt)
    
    var hardNegZ: Option[Tensor[ParamType]] = None
    if (hardNegative) {
      val (_, _, z3, _, _, _) = predict(qProcessed, sFlip, pidOpt)
      hardNegZ = Some(z3)
    }
    
    // CL loss calculation
    val lens = (sProcessed >= 0).sum(dim = 1)
    val minlen = lens.min().int()
    
    val input = sim(z1.slice(1, 0, minlen), z2.slice(1, 0, minlen))
    
    val inputWithHardNeg = hardNegZ.map {
      z3 => torch.cat(Seq(input, sim(z1.slice(1, 0, minlen), z3.slice(1, 0, minlen))), dim = 1)
    }.getOrElse(input)
    
    val target = torch.arange(sProcessed.shape(0))
      .unsqueeze(1)
      .expand(-1, minlen)
    
    val clLoss = F.cross_entropy(inputWithHardNeg, target)
    
    // Prediction loss
    var predLoss = torch.zeros()
    for (i <- 1 until windowSize) {
      val label = sProcessed.slice(1, i, sProcessed.shape(1))
      val query = qEmb.slice(1, i, qEmb.shape(1))
      val h = readout(z1.slice(1, 0, query.shape(1)), query)
      
      val y = if (trans) {
        out(torch.cat(Seq(query, h), dim = -1))
      } else {
        out(torch.cat(Seq(query, h), dim = -1)).squeeze(-1)
      }
      
      val maskedY = y.where(label >= 0, y, Tensor.empty[ParamType]())
      val maskedLabel = label.where(label >= 0, label, Tensor.empty[ParamType]())
      predLoss = predLoss + F.binarycross_entropyWithLogits(maskedY, maskedLabel, reduction = "mean")
    }
    
    // Calculate final predictions
    val m = nn.Sigmoid()
    val preds = m(logits)
    
    val (finalPreds, trueLabels) = if (trans) {
      val truePreds = (preds * F.oneHot(cshft.get.toLong, numSkills)).sum(dim = -1)
      val trueTargets = r.slice(1, length, r.shape(1))
      (truePreds, trueTargets)
    } else if (maskFuture || predLast || maskResponse) {
      val predSliced = preds.slice(1, preds.shape(1) - length, preds.shape(1))
      val trueSliced = r.slice(1, r.shape(1) - length, r.shape(1))
      (predSliced, trueSliced)
    } else {
      val predSliced = preds.slice(1, length, preds.shape(1))
      val trueSliced = r.slice(1, length, r.shape(1))
      (predSliced, trueSliced)
    }
    
    Map(
      "pred" -> finalPreds,
      "true" -> trueLabels,
      "reg_loss" -> (clLoss * lambdaClValue + regLoss)
    )
  }
  
  // Loss calculation
  def loss(feedDict: Map[String, Tensor[ParamType]], outDict: Map[String, Tensor[ParamType]]): 
      (Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]) = {
    val pred = outDict("pred").flatten()
    val trueTensor = outDict("true").flatten()
    val regLoss = outDict("reg_loss")
    val mask = trueTensor.gt(-1)
    
    val loss = lossFn(pred(mask), trueTensor(mask))
    
    (loss + regLoss, mask.sum(), trueTensor(mask).sum())
  }
  
  // Similarity calculation for contrastive learning
  def sim(z1: Tensor[ParamType], z2: Tensor[ParamType]): Tensor[ParamType] = {
    val bs = z1.shape(0)
    val seqlen = z1.shape(1)
    
    var z1Processed = z1.unsqueeze(1).view(bs, 1, seqlen, nKnow, -1)
    var z2Processed = z2.unsqueeze(0).view(1, bs, seqlen, nKnow, -1)
    
    projLayer.foreach {
      proj =>
        z1Processed = proj(z1Processed)
        z2Processed = proj(z2Processed)
    }
    
    F.cosineSimilarity(z1Processed.mean(dim = 3), z2Processed.mean(dim = 3), dim = -1) / 0.05f
  }
  
  // Apply method for TensorModule
  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = ???
  
  // Apply method with dictionary input
  def apply(feedDict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = getClLoss(feedDict)
}

class DTransformerLayer[ParamType <: FloatNN: Default](
    embeddingSize: Int,
    nHeads: Int,
    dropout: Float,
    kqSame: Boolean = true
) extends TensorModule[ParamType] with HasParams[ParamType] {
  
  val maskedAttnHead = new MultiHeadAttention[ParamType](embeddingSize, nHeads, kqSame)
  val dropoutRate = dropout
  val dropoutLayer = nn.Dropout(dropout)
  val layerNorm = nn.LayerNorm(embeddingSize)
  
  // Collect parameters
  override val params: Seq[Tensor[ParamType]] = {
    maskedAttnHead.params ++ 
    layerNorm.params
  }
  
  // Get device
  def device(): Device = {
    params.headOption.map(_.device).getOrElse(CPU)
  }
  
  // Forward method
  def forward(query: Tensor[ParamType], key: Tensor[ParamType], values: Tensor[ParamType], 
              lens: Tensor[ParamType], peekCur: Boolean = false): 
      (Tensor[ParamType], Tensor[ParamType]) = {
    val seqlen = query.shape(1)
    
    // Construct mask
    val mask = torch.ones(Seq(seqlen, seqlen)).tril(if (peekCur) 0 else -1)
    val boolMask = mask.bool().unsqueeze(0).unsqueeze(0).to(device())
    
    // Apply transformer layer
    val (queryUpdated, scores) = maskedAttnHead(query, key, values, boolMask, maxout = !peekCur)
    val result = layerNorm(query + dropoutLayer(queryUpdated))
    
    (result, scores)
  }
  
  // Apply method
  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = ???
  
  // Apply method with full parameters
  def apply(query: Tensor[ParamType], key: Tensor[ParamType], values: Tensor[ParamType], 
            lens: Tensor[ParamType], peekCur: Boolean = false): 
      (Tensor[ParamType], Tensor[ParamType]) = forward(query, key, values, lens, peekCur)
}

class MultiHeadAttention3[ParamType <: FloatNN: Default](
    embeddingSize: Int,
    nHeads: Int,
    kqSame: Boolean = true,
    bias: Boolean = true
) extends TensorModule[ParamType] with HasParams[ParamType] {
  
  val dK = embeddingSize / nHeads
  val h = nHeads
  
  val qLinear = nn.Linear(embeddingSize, embeddingSize, bias = bias)
  val kLinear = if (kqSame) qLinear else nn.Linear(embeddingSize, embeddingSize, bias = bias)
  val vLinear = nn.Linear(embeddingSize, embeddingSize, bias = bias)
  
  val outProj = nn.Linear(embeddingSize, embeddingSize, bias = bias)
  val gammas = torch.zeros(Seq(nHeads, 1, 1))
  nn.init.xavierUniform(gammas)
  
  // Collect parameters
  override val params: Seq[Tensor[ParamType]] = {
    val paramsList = ListBuffer[Tensor[ParamType]]()
    paramsList.append(qLinear.weight)
    if (qLinear.bias.isDefined) paramsList.append(qLinear.bias.get)
    
    if (!kqSame) {
      paramsList.append(kLinear.weight)
      if (kLinear.bias.isDefined) paramsList.append(kLinear.bias.get)
    }
    
    paramsList.append(vLinear.weight)
    if (vLinear.bias.isDefined) paramsList.append(vLinear.bias.get)
    
    paramsList.append(outProj.weight)
    if (outProj.bias.isDefined) paramsList.append(outProj.bias.get)
    
    paramsList.append(gammas)
    
    paramsList.toSeq
  }
  
  // Forward method
  def forward(q: Tensor[ParamType], k: Tensor[ParamType], v: Tensor[ParamType], 
              mask: Tensor[ParamType], maxout: Boolean = false): 
      (Tensor[ParamType], Tensor[ParamType]) = {
    val bs = q.shape(0)
    
    // Perform linear operation and split into h heads
    var qProcessed = qLinear(q).view(bs, -1, h, dK)
    var kProcessed = kLinear(k).view(bs, -1, h, dK)
    var vProcessed = vLinear(v).view(bs, -1, h, dK)
    
    // Transpose to get dimensions bs * h * sl * d_k
    kProcessed = kProcessed.transpose(1, 2)
    qProcessed = qProcessed.transpose(1, 2)
    vProcessed = vProcessed.transpose(1, 2)
    
    // Calculate attention
    val (vOutput, scores) = attention(qProcessed, kProcessed, vProcessed, mask, Some(gammas), maxout)
    
    // Concatenate heads and put through final linear layer
    val concat = vOutput.transpose(1, 2).contiguous().view(bs, -1, embeddingSize)
    val output = outProj(concat)
    
    (output, scores)
  }
  
  // Apply method
  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = ???
  
  // Apply method with full parameters
  def apply(q: Tensor[ParamType], k: Tensor[ParamType], v: Tensor[ParamType], 
            mask: Tensor[ParamType], maxout: Boolean = false): 
      (Tensor[ParamType], Tensor[ParamType]) = forward(q, k, v, mask, maxout)
}

// Attention function
def attention[ParamType <: FloatNN: Default](
    q: Tensor[ParamType],
    k: Tensor[ParamType],
    v: Tensor[ParamType],
    mask: Tensor[ParamType],
    gamma: Option[Tensor[ParamType]] = None,
    maxout: Boolean = false
): (Tensor[ParamType], Tensor[ParamType]) = {
  val dK = k.shape(-1)
  var scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dK).toFloat
  val (bs, head, seqlen, _) = (scores.shape(0), scores.shape(1), scores.shape(2), scores.shape(3))
  
  // Include temporal effect
  gamma.foreach {
    gammaTensor =>
      val x1 = torch.arange(seqlen).expand(seqlen, -1).to(gammaTensor.device)
      val x2 = x1.transpose(0, 1).contiguous()
      
      // Calculate position effect
      val scoresMasked = scores.masked_fill(mask == 0, -1e32f)
      val softmaxScores = F.softmax(scoresMasked, dim = -1)
      
      val distCumScores = softmaxScores.cumsum(dim = -1)
      val distTotalScores = softmaxScores.sum(dim = -1, keepdim = true)
      val positionEffect = x1.sub(x2).abs().unsqueeze(0).unsqueeze(0)
      
      val distScores = (distTotalScores - distCumScores) * positionEffect
      val clampedDistScores = distScores.clamp(min = 0.0f).sqrt().detach()
      
      val gammaAbs = -1.0f * gammaTensor.abs().unsqueeze(0)
      val totalEffect = (clampedDistScores * gammaAbs).exp().clamp(min = 1e-5f, max = 1e5f)
      
      scores = scores * totalEffect
  }
  
  // Normalize attention score
  scores = scores.masked_fill(mask == 0, -1e32f)
  scores = F.softmax(scores, dim = -1)
  scores = scores.masked_fill(mask == 0, 0.0f) // Set to hard zero to avoid leakage
  
  // Max-out scores
  if (maxout) {
    val maxScores = scores.max(dim = -1, keepdim = true)._1
    val scale = (1.0f / (maxScores + 1e-8f)).clamp(max = 5.0f)
    scores = scores * scale
  }
  
  // Calculate output
  val output = torch.matmul(scores, v)
  
  (output, scores)
}

// Companion object for DTransformer
object DTransformer {
  def apply[ParamType <: FloatNN: Default](
      maskResponse: Boolean,
      predLast: Boolean,
      maskFuture: Boolean,
      length: Int,
      trans: Boolean,
      numSkills: Int,
      numQuestions: Int,
      embeddingSize: Int = 64,
      dFF: Int = 256,
      numAttnHeads: Int = 8,
      nKnow: Int = 16,
      numBlocks: Int = 3,
      dropout: Float = 0.3f,
      lambdaCl: Float = 0.1f,
      proj: Boolean = false,
      hardNeg: Boolean = false,
      window: Int = 1,
      shortcut: Boolean = false,
      separateQr: Boolean = false
  ): DTransformer[ParamType] = {
    new DTransformer[ParamType](
      maskResponse, predLast, maskFuture, length, trans, numSkills, numQuestions,
      embeddingSize, dFF, numAttnHeads, nKnow, numBlocks, dropout,
      lambdaCl, proj, hardNeg, window, shortcut, separateQr
    )
  }
}
