package ru.ifmo.rain.algorithms.spectral

import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{VectorUDT, DenseVector => OldDense, SparseVector => OldSparse}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK
import ru.ifmo.rain.algorithms.ClusteringAlgo
import ru.ifmo.rain.algorithms.kmeans.KMeansLocal
import ru.ifmo.rain.algorithms.spectral.EigenValueDecomposition.symmetricEigs
import ru.ifmo.rain.distances.Cosine.dot
import ru.ifmo.rain.distances.{ModalWrapper, MultiBuilder, MultiDistance}
import ru.ifmo.rain.{ID_COL, ID_FIELD, LABEL_COL, withTime}


abstract class SpectralClustering(
                                   val eigen: Int,
                                   val k: Int,
                                   val maxIterations: Int,
                                   val seed: Long
                                 ) extends ClusteringAlgo[SpectralModel] {

  override def verifyParams: Option[String] = {
    if (eigen < 2) Option("eigen")
    else if (k < 2) Option("k")
    else if (maxIterations < 1) Option("maxIterations")
    else Option.empty
  }

  def fit(df: DataFrame, dist: MultiDistance): SpectralModel = withTime("Spectral", clear = true) {
    val modalGraphs = withTime("Graphs") { calcModalGraphs(df, dist) }
    val spectralEmbeddings = withTime("Spectral Embeddings") { calcSpectralEmbeddings(modalGraphs) }

    val eigenDf = getEigenDataFrame(spectralEmbeddings).toArray
    val eigenDist = transformDistance(eigenDf, dist)

    val labeledCentroids = withTime("Local KMeans") {
      val kMeans = new KMeansLocal(k, maxIterations, seed)
      val centroids = kMeans(eigenDf, eigenDist, Array.fill(eigenDf.length)(1.0))
      centroids.zipWithIndex
    }

    val ss = df.sparkSession
    val broadcastCentroids = ss.sparkContext.broadcast(labeledCentroids)
    val clusteredData = eigenDf.zipWithIndex.map { case (eigenObj, id) =>
      val distances = broadcastCentroids.value.map { c => c._2 -> eigenDist(eigenObj, c._1) }
      (id.toLong, distances.minBy(_._2)._1)
    }
    val labels = ss.createDataFrame(clusteredData)
      .withColumnRenamed("_1", ID_COL)
      .withColumnRenamed("_2", LABEL_COL)
    val clustered = df.join(labels, ID_COL)
    clustered.persist(MEMORY_AND_DISK).count()
    new SpectralModel(clustered)
  }

  protected def modalGraph(modality: RDD[(Long, Vector)], n: Int, distance: (Vector, Vector) => Double): RowMatrix

  private def calcModalGraphs(df: DataFrame, dist: MultiDistance): Array[(ModalWrapper, RowMatrix)] = {
    dist.modalities.map { wrapper =>
      val data = df.rdd.map { row => row.getAs[Long](ID_COL) -> row.getAs[Vector](wrapper.modality) }
      val n = data.persist(MEMORY_AND_DISK).count().toInt
      val matrix = modalGraph(data, n, wrapper.apply)
      val rows = matrix.rows.persist(MEMORY_AND_DISK).count()
      assert(rows.toInt == n, "Matrix rows count is not the same as data count")
      data.unpersist()
      wrapper -> matrix
    }
  }

  private def calcSpectralEmbeddings(modalGraphs: Array[(ModalWrapper, RowMatrix)]): Array[(ModalWrapper, Array[DenseVector])] =
    modalGraphs.map { case (wrapper, matrix) => wrapper -> spectralEmbedding(matrix, eigen)._2 }

  private def spectralEmbedding(matrix: RowMatrix, k: Int): (Array[Double], Array[DenseVector]) = {
    val n = matrix.numRows().toInt
    val iterations = math.max(300, k * 3)
    val (values, vectors) = symmetricEigs(eigenMul(matrix), n, k, 1e-10, iterations)

    val eigenModal = for (row <- 0 until vectors.rows) yield {
      val vector = for (col <- 0 until vectors.cols) yield vectors(row, col)
      new DenseVector(vector.toArray)
    }
    values.toArray -> eigenModal.toArray
  }

  private type BreezeMul = BDV[Double] => BDV[Double]
  private def eigenMul(matrix: RowMatrix): BreezeMul = v => {
    val sv = new DenseVector(v.toArray)
    val svBroadcast = matrix.rows.context.broadcast(sv)
    val values = matrix.rows.map {
      case dr: OldDense => dot(dr.asML, svBroadcast.value)
      case sr: OldSparse => dot(svBroadcast.value, sr.asML)
    }.collect()
    new BDV[Double](values)
  }

  private def getEigenDataFrame(spectralEmbeddings: Array[(ModalWrapper, Array[DenseVector])]): IndexedSeq[Row] = {
    val eigenFields = spectralEmbeddings.map { case (wrapper, _) =>
      StructField(wrapper.modality, new VectorUDT, nullable = false)
    }.toList
    val size = spectralEmbeddings.head._2.length
    val eigenSchema = new StructType((ID_FIELD :: eigenFields).toArray)
    for (id <- 0 until size) yield {
      val obj = spectralEmbeddings.map { case (_, rows) => rows(id) }.toList
      new GenericRowWithSchema((id.toLong :: obj).toArray, eigenSchema)
    }
  }

  private def transformDistance(eigenDf: Array[Row], dist: MultiDistance): MultiDistance = {
    val wrappers = dist.modalities.map { wrapper =>
      val norm = MultiBuilder.modalityNorm(eigenDf, wrapper.modality, wrapper.fun)
      new ModalWrapper(wrapper.fun, wrapper.modality, eigen, wrapper.weight, norm)
    }
    new MultiDistance(wrappers.sortBy(_.modality))
  }
}