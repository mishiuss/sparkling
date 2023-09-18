package ru.ifmo.rain.algorithms.bisecting

import org.apache.spark.sql.Row
import ru.ifmo.rain.distances.MultiDistance

import scala.annotation.tailrec

class ClusteringTreeNode (
                           val index: Int,
                           val size: Long,
                           val center: Row,
                           val cost: Double,
                           val height: Double,
                           val children: Array[ClusteringTreeNode]
                         ) extends Serializable {

  private val isLeaf: Boolean = children.isEmpty

  require((isLeaf && index >= 0) || (!isLeaf && index < 0))

  @tailrec
  final def predict(obj: Row, cost: Double)(implicit dist: MultiDistance): (Int, Double) = {
    if (isLeaf) {
      (index, cost)
    } else {
      val (selectedChild, minCost) = children.map { child =>
        child -> dist(child.center, obj)
      }.minBy(_._2)
      selectedChild.predict(obj, minCost)
    }
  }

  def leafNodes: Array[ClusteringTreeNode] =
    if (isLeaf) Array(this)
    else children.flatMap(_.leafNodes)
}