package ru.ifmo.rain.algorithms.spectral

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import org.apache.spark.rdd.RDD
import ru.ifmo.rain.Sparkling

import scala.math.exp


@Sparkling
class SpectralSimilarity(
                          val gamma: Double,
                          eigen: Int,
                          k: Int,
                          maxIterations: Int,
                          seed: Long
                        ) extends SpectralClustering(eigen, k, maxIterations, seed) {

  override def verifyParams: Option[String] = {
    if (gamma < 0.0) Option("gamma")
    else super.verifyParams
  }

  override protected def modalGraph(modality: RDD[(Long, Vector)], n: Int, distance: (Vector, Vector) => Double): RowMatrix = {
    val g = gamma
    val affinity = modality.cartesian(modality).map { case ((xId, x), (yId, y)) =>
      val d = distance(x, y)
      xId -> (yId.toInt, exp(-g * d * d))
    }
    val rows = affinity.groupByKey().map { case (id, idxValues) =>
      val sortedByIdxValues = idxValues.toArray.sortBy(_._1)
      id -> OldVectors.dense(sortedByIdxValues.map(_._2))
    }.sortBy(_._1)

    new RowMatrix(rows.map(_._2), n.toLong, n)
  }
}