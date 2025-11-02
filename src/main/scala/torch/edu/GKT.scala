package torch.edu

import torch.*
import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn as nn
import torch.nn.*
import scala.collection.mutable.ListBuffer

class GKT[ParamType <: FloatNN: Default](
    length: Int,
    numSkills: Int,
    graph: Tensor[ParamType],
    hiddenDim: Int,
    embeddingSize: Int,
    graphType: String = "dense",
    dropout: Float = 0.5f
) extends TensorModule[ParamType] with HasParams[ParamType] {
  
  val resLen = 2
  val graphParam = torch.tensor(graph.data).requiresGrad_(False)
  
  // One-hot feature and question embeddings
  val oneHotFeat = torch.eye(resLen * numSkills)
  val oneHotQ = torch.cat(
    Seq(
      torch.eye(numSkills),
      torch.zeros(Seq(1, numSkills))
    ),
    dim = 0
  )
  
  // Concept and interaction embeddings
  val interactionEmb = nn.Embedding(resLen * numSkills, embeddingSize)
  val embC = nn.Embedding(numSkills + 1, embeddingSize, paddingIdx = -1)
  
  // MLP layers
  val mlpInputDim = hiddenDim + embeddingSize
  val fSelf = new MLP[ParamType](mlpInputDim, hiddenDim, hiddenDim, dropout, bias = true)
  
  // f_neighbor functions
  val fNeighborList = ListBuffer[MLP[ParamType]]()
  fNeighborList.append(new MLP[ParamType](2 * mlpInputDim, hiddenDim, hiddenDim, dropout, bias = true))
  fNeighborList.append(new MLP[ParamType](2 * mlpInputDim, hiddenDim, hiddenDim, dropout, bias = true))
  
  // Erase & Add Gate
  val eraseAddGate = new EraseAddGate[ParamType](hiddenDim, numSkills)
  
  // GRU
  val gru = nn.GRUCell(hiddenDim, hiddenDim, bias = true)
  
  // Prediction layer
  val predict = nn.Linear(hiddenDim, 1, bias = true)
  val lossFn = nn.BCELoss(reduction = "mean")
  
  // Collect all parameters
  override val params: Seq[Tensor[ParamType]] = {
    val paramsList = ListBuffer[Tensor[ParamType]]()
    paramsList.appendAll(interactionEmb.params)
    paramsList.appendAll(embC.params)
    paramsList.appendAll(fSelf.params)
    fNeighborList.foreach(mlp => paramsList.appendAll(mlp.params))
    paramsList.appendAll(eraseAddGate.params)
    paramsList.appendAll(gru.params)
    paramsList.appendAll(predict.params)
    paramsList.toSeq
  }
  
  // Aggregate step
  def aggregate(xt: Tensor[ParamType], qt: Tensor[ParamType], ht: Tensor[ParamType], batchSize: Int): Tensor[ParamType] = {
    val qtMask = qt.ne(-1)
    val xIdxMat = torch.arange(resLen * numSkills)
    val xEmbedding = interactionEmb(xIdxMat.toLong)
    
    val maskedFeat = F.embedding(xt(qtMask).toLong, oneHotFeat)
    val resEmbedding = maskedFeat.matmul(xEmbedding)
    val maskNum = resEmbedding.shape(0)
    
    val conceptIdxMat = Tensor.full[ParamType](Seq(batchSize, numSkills), numSkills.toFloat).toLong
    val qtMaskExpanded = qtMask.unsqueeze(1).expand(batchSize, numSkills)
    val arangeMat = torch.arange(numSkills).unsqueeze(0).expand(batchSize, numSkills).toLong
    conceptIdxMat.where(qtMaskExpanded, arangeMat, conceptIdxMat)
    
    val conceptEmbedding = embC(conceptIdxMat)
    
    val qtMasked = qt(qtMask).toLong
    val indices = Seq(
      torch.arange(maskNum).toLong,
      qtMasked
    )
    conceptEmbedding.indexPut(indices, resEmbedding)
    
    torch.cat(Seq(ht, conceptEmbedding), dim = -1)
  }
  
  // GNN aggregation step
  def aggNeighbors(tmpHt: Tensor[ParamType], qt: Tensor[ParamType]): (Tensor[ParamType], Option[Tensor[ParamType]], Option[Tensor[ParamType]], Option[Tensor[ParamType]]) = {
    val qtMask = qt.ne(-1)
    val maskedQt = qt(qtMask).toLong
    val maskedTmpHt = tmpHt(qtMask)
    val maskNum = maskedTmpHt.shape(0)
    
    val indices = Seq(
      torch.arange(maskNum).toLong,
      maskedQt
    )
    val selfHt = maskedTmpHt(indices)
    val selfFeatures = fSelf(selfHt)
    
    val expandedSelfHt = selfHt.unsqueeze(1).expand(maskNum, numSkills, mlpInputDim)
    val neighHt = torch.cat(Seq(expandedSelfHt, maskedTmpHt), dim = -1)
    
    val adj = graphParam(maskedQt).unsqueeze(-1)
    val reverseAdj = graphParam.transpose(0, 1)(maskedQt).unsqueeze(-1)
    
    val neighFeatures = adj * fNeighborList(0)(neighHt) + reverseAdj * fNeighborList(1)(neighHt)
    
    val mNext = tmpHt.slice(-1, 0, hiddenDim)
    mNext.indexPut(Seq(qtMask), neighFeatures)
    mNext.indexPut(Seq(qtMask) ++ indices, selfFeatures)
    
    (mNext, None, None, None)
  }
  
  // Update step
  def update(tmpHt: Tensor[ParamType], ht: Tensor[ParamType], qt: Tensor[ParamType]): (Tensor[ParamType], Option[Tensor[ParamType]], Option[Tensor[ParamType]], Option[Tensor[ParamType]]) = {
    val qtMask = qt.ne(-1)
    val (mNext, conceptEmbedding, recEmbedding, zProb) = aggNeighbors(tmpHt, qt)
    
    mNext.indexPut(Seq(qtMask), eraseAddGate(mNext(qtMask)))
    
    val hNext = mNext
    val res = gru(
      mNext(qtMask).reshape(-1, hiddenDim),
      ht(qtMask).reshape(-1, hiddenDim)
    )
    
    hNext.indexPut(
      Seq(qtMask), 
      res.reshape(-1, numSkills, hiddenDim)
    )
    
    (hNext, conceptEmbedding, recEmbedding, zProb)
  }
  
  // Predict step
  def predictStep(hNext: Tensor[ParamType], qt: Tensor[ParamType]): Tensor[ParamType] = {
    val qtMask = qt.ne(-1)
    val y = predict(hNext).squeeze(-1)
    y.where(qtMask, F.sigmoid(y), y)
    
    y
  }
  
  // Get next prediction
  def getNextPred(yt: Tensor[ParamType], qNext: Tensor[ParamType]): Tensor[ParamType] = {
    val nextQt = qNext.where(qNext.ne(-1), torch.full_like(qNext, numSkills.toFloat))
    val oneHotQt = F.embedding(nextQt.toLong, oneHotQ)
    (yt * oneHotQt).sum(dim = 1)
  }
  
  // Forward method
  def forward(feedDict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = {
    val q = feedDict("skills")
    val r = feedDict("responses")
    val maskedR = r * r.gt(-1)
    val features = q * 2 + maskedR
    val questions = q
    
    val batchSize = features.shape(0)
    val seqLen = features.shape(1)
    val device = q.device
    
    var ht = torch.zeros(Seq(batchSize, numSkills, hiddenDim), device = device)
    val predList = ListBuffer[Tensor[ParamType]]()
    
    for (i <- 0 until seqLen) {
      val xt = features.slice(1, i, i + 1).squeeze(1)
      val qt = questions.slice(1, i, i + 1).squeeze(1)
      val qtMask = qt.ne(-1)
      
      val tmpHt = aggregate(xt, qt, ht, batchSize)
      val (hNext, _, _, _) = update(tmpHt, ht, qt)
      
      ht.where(qtMask.unsqueeze(1).unsqueeze(2).expand(batchSize, numSkills, hiddenDim), hNext, ht)
      
      val yt = predictStep(hNext, qt)
      
      if (i < seqLen - 1) {
        val pred = getNextPred(yt, questions.slice(1, i + 1, i + 2).squeeze(1))
        predList.append(pred)
      }
    }
    
    val predRes = Tensor.stack(predList.toSeq, dim = 1)
    val outDict = Map(
      "pred" -> predRes.slice(1, length - 1, predRes.shape(1)),
      "true" -> r.slice(1, length, r.shape(1))
    )
    
    outDict
  }
  
  // Loss method
  def loss(feedDict: Map[String, Tensor[ParamType]], outDict: Map[String, Tensor[ParamType]]): (Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]) = {
    val pred = outDict("pred").flatten()
    val trueTensor = outDict("true").flatten()
    val mask = trueTensor.gt(-1)
    
    val loss = lossFn(pred(mask), trueTensor(mask))
    
    (loss, mask.sum(), trueTensor(mask).sum())
  }
  
  // Apply method
  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = ???
  
  // Apply method with dictionary input
  def apply(feedDict: Map[String, Tensor[ParamType]]): Map[String, Tensor[ParamType]] = forward(feedDict)
}

class MLP[ParamType <: FloatNN: Default](
    inputDim: Int,
    hiddenDim: Int,
    outputDim: Int,
    dropout: Float = 0f,
    bias: Boolean = true
) extends TensorModule[ParamType] with HasParams[ParamType] {
  
  val fc1 = nn.Linear(inputDim, hiddenDim, bias = bias)
  val fc2 = nn.Linear(hiddenDim, outputDim, bias = bias)
  val norm = nn.BatchNorm1d(outputDim)
  val dropoutRate = dropout
  
  // Initialize weights
  initWeights()
  
  def initWeights(): Unit = {
    fc1.weight.data = nn.init.xavierNormal(fc1.weight.data.shape)
    if (bias && fc1.bias.isDefined) {
      fc1.bias.get.data.fill_(0.1f)
    }
    
    fc2.weight.data = nn.init.xavierNormal(fc2.weight.data.shape)
    if (bias && fc2.bias.isDefined) {
      fc2.bias.get.data.fill_(0.1f)
    }
    
    norm.weight.data.fill_(1f)
    if (norm.bias.isDefined) {
      norm.bias.get.data.fill_(0f)
    }
  }
  
  // Batch normalization helper
  def batchNorm(inputs: Tensor[ParamType]): Tensor[ParamType] = {
    if (inputs.numel() == outputDim || inputs.numel() == 0) {
      inputs
    } else if (inputs.dim() == 3) {
      val x = inputs.reshape(inputs.shape(0) * inputs.shape(1), -1)
      val normalized = norm(x)
      normalized.reshape(inputs.shape(0), inputs.shape(1), -1)
    } else {
      norm(inputs)
    }
  }
  
  // Forward method
  def forward(inputs: Tensor[ParamType]): Tensor[ParamType] = {
    var x = F.relu(fc1(inputs))
    x = F.dropout(x, dropoutRate)
    x = F.relu(fc2(x))
    batchNorm(x)
  }
  
  // Collect parameters
  override val params: Seq[Tensor[ParamType]] = {
    val paramsList = ListBuffer[Tensor[ParamType]]()
    paramsList.append(fc1.weight)
    if (fc1.bias.isDefined) paramsList.append(fc1.bias.get)
    paramsList.append(fc2.weight)
    if (fc2.bias.isDefined) paramsList.append(fc2.bias.get)
    paramsList.append(norm.weight)
    if (norm.bias.isDefined) paramsList.append(norm.bias.get)
    paramsList.toSeq
  }
  
  // Apply method
  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
}

class EraseAddGate[ParamType <: FloatNN: Default](
    featureDim: Int,
    numSkills: Int,
    bias: Boolean = true
) extends TensorModule[ParamType] with HasParams[ParamType] {
  
  val weight = Tensor.randn[ParamType](Seq(numSkills))
  resetParameters()
  
  val erase = nn.Linear(featureDim, featureDim, bias = bias)
  val add = nn.Linear(featureDim, featureDim, bias = bias)
  
  def resetParameters(): Unit = {
    val stdv = 1.0f / math.sqrt(weight.shape(0)).toFloat
    weight.data = Tensor.rand[ParamType](weight.data.shape, min = -stdv, max = stdv)
  }
  
  // Forward method
  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    val eraseGate = F.sigmoid(erase(x))
    val tmpX = x - weight.unsqueeze(1) * eraseGate * x
    val addFeat = F.tanh(add(x))
    tmpX + weight.unsqueeze(1) * addFeat
  }
  
  // Collect parameters
  override val params: Seq[Tensor[ParamType]] = {
    val paramsList = ListBuffer[Tensor[ParamType]]()
    paramsList.append(weight)
    paramsList.append(erase.weight)
    if (erase.bias.isDefined) paramsList.append(erase.bias.get)
    paramsList.append(add.weight)
    if (add.bias.isDefined) paramsList.append(add.bias.get)
    paramsList.toSeq
  }
  
  // Apply method
  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
}

// Companion object for GKT
object GKT {
  def apply[ParamType <: FloatNN: Default](
      length: Int,
      numSkills: Int,
      graph: Tensor[ParamType],
      hiddenDim: Int,
      embeddingSize: Int,
      graphType: String = "dense",
      dropout: Float = 0.5f
  ): GKT[ParamType] = {
    new GKT[ParamType](length, numSkills, graph, hiddenDim, embeddingSize, graphType, dropout)
  }
}
