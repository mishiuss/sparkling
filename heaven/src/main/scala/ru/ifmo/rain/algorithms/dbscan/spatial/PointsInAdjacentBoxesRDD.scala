package ru.ifmo.rain.algorithms.dbscan.spatial

import org.apache.spark.rdd.{RDD, ShuffledRDD}
import ru.ifmo.rain.algorithms.dbscan.box.{MultiModalBox, BoxCalculator}
import ru.ifmo.rain.algorithms.dbscan.point.Point


class PointsInAdjacentBoxesRDD(prev: RDD[((Int, Int), Point)], val adjacentBoxIdPairs: Array[(Int, Int)])
  extends ShuffledRDD[(Int, Int), Point, Point](prev, new AdjacentBoxesPartitioner(adjacentBoxIdPairs))


object PointsInAdjacentBoxesRDD {
  def apply(points: RDD[Point], boxesWithAdjacentBoxes: Iterable[MultiModalBox]): PointsInAdjacentBoxesRDD = {
    val adjacentBoxIdPairs = BoxCalculator.generateDistinctPairsOfAdjacentBoxIds(boxesWithAdjacentBoxes).toArray
    val broadcastBoxIdPairs = points.sparkContext.broadcast(adjacentBoxIdPairs)
    val pointsKeyedByPairOfBoxes = points.mapPartitions(it => {
      val boxIdPairs = broadcastBoxIdPairs.value
      for (pt <- it; (p1, p2) <- boxIdPairs; if pt.boxId == p1 || pt.boxId == p2) yield ((p1, p2), pt)
    })
    new PointsInAdjacentBoxesRDD(pointsKeyedByPairOfBoxes, adjacentBoxIdPairs)
  }
}