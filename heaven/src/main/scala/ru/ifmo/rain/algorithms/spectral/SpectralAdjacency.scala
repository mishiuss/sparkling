package ru.ifmo.rain.algorithms.spectral

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import org.apache.spark.rdd.RDD
import ru.ifmo.rain.Sparkling

import scala.collection.mutable


@Sparkling
class SpectralAdjacency(
                         val neighbours: Int,
                         eigen: Int,
                         k: Int,
                         maxIterations: Int,
                         seed: Long
                       ) extends SpectralClustering(eigen, k, maxIterations, seed) {

  override def verifyParams: Option[String] = {
    if (neighbours < 1) Option("neighbours")
    else super.verifyParams
  }

  override protected def modalGraph(modality: RDD[(Long, Vector)], n: Int, distance: (Vector, Vector) => Double): RowMatrix = {
    val numNeighbours = neighbours
    val affinity = modality.cartesian(modality).map { case ((xId, x), (yId, y)) => xId -> (yId.toInt, distance(x, y)) }
    val zeroSet = mutable.TreeSet[(Int, Double)]()(Ordering.by(_._2))

    val neighboursIds = affinity.aggregateByKey(zeroSet)(
      seqOp = (nearest, idxValue) => {
        if (nearest.size <= numNeighbours) nearest += idxValue
        else if (nearest.last._2 > idxValue._2) nearest.dropRight(1) += idxValue
        else nearest
      },
      combOp = (s1, s2) => (s1 ++ s2).take(numNeighbours + 1)
    )

    val laplacian = neighboursIds.map { case (rowId, nearest) =>
      val values = nearest.map { case (id, _) =>
        id -> (if (id == rowId) numNeighbours.toDouble else -1.0)
      }
      rowId -> OldVectors.sparse(n, values.toSeq)
    }.sortBy(_._1)

    new RowMatrix(laplacian.map(_._2), n.toLong, n)
  }
}