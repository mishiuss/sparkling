package ru.ifmo.rain.algorithms.dbscan

import org.apache.spark.rdd.{RDD, ShuffledRDD}
import ru.ifmo.rain.algorithms.dbscan.box.MultiModalBox
import ru.ifmo.rain.algorithms.dbscan.point.{Point, PointSortKey}
import ru.ifmo.rain.algorithms.dbscan.spatial.{BoxPartitioner, PartitionIndex, PointsPartitionedByBoxesRDD}
import ru.ifmo.rain.distances.MultiDistance

import scala.collection.mutable


class DistanceAnalyzer(implicit settings: Settings, distance: MultiDistance) extends Serializable {
  def countNeighborsForEachPoint(data: PointsPartitionedByBoxesRDD): RDD[(PointSortKey, Point)] = {
    val closePointCounts = countClosePoints(data).foldByKey(1) { _ + _ }.cache()
    val pointsWithoutNeighbors = data.keys.subtract(closePointCounts.keys).map { x => (x, 1L) }

    val allPointCounts = closePointCounts.union(pointsWithoutNeighbors)
    val partitionedAndSortedCounts = new ShuffledRDD[PointSortKey, Long, Long](
      allPointCounts, new BoxPartitioner(data.boxes)
    ).mapPartitions(sortPartition, preservesPartitioning = true)

    data.mapPartitions(sortPartition, preservesPartitioning = true)
      .zipPartitions(partitionedAndSortedCounts, preservesPartitioning = true)((it1, it2) => {
        it1.zip(it2).map(x => {
          assert(x._1._1.pointId == x._2._1.pointId)
          val newPt = new Point(x._1._2).withNumberOfNeighbors(x._2._2)
          (new PointSortKey(newPt), newPt)
        })
      })
  }

  private def sortPartition[T](it: Iterator[(PointSortKey, T)]): Iterator[(PointSortKey, T)] = {
    val ordering = implicitly[Ordering[PointSortKey]]
    it.toArray.sortWith((x, y) => ordering.lt(x._1, y._1)).iterator
  }

  private def countClosePoints(data: PointsPartitionedByBoxesRDD): RDD[(PointSortKey, Long)] = {
    val closePointsInsideBoxes = countClosePointsWithinEachBox(data)
    val pointsCloseToBoxBounds = findPointsCloseToBoxBounds(data, data.boxes)
    val closePointsInDifferentBoxes = countClosePointsInDifferentBoxes(pointsCloseToBoxBounds, data.boxes)
    closePointsInsideBoxes.union(closePointsInDifferentBoxes)
  }

  private def countClosePointsWithinEachBox(data: PointsPartitionedByBoxesRDD): RDD[(PointSortKey, Long)] = {
    val broadcastBoxes = data.sparkContext.broadcast(data.boxes)
    data.mapPartitionsWithIndex((partitionIndex, it) => {
      val boundingBox = broadcastBoxes.value.find { _.partitionId == partitionIndex }.get
      countClosePointsWithinPartition(it, boundingBox)
    })
  }

  private def countClosePointsWithinPartition(it: Iterator[(PointSortKey, Point)], boundingBox: MultiModalBox): Iterator[(PointSortKey, Long)] = {
    val (it1, it2) = it.duplicate
    val partitionIndex = new PartitionIndex(boundingBox)
    val counts = mutable.HashMap[PointSortKey, Long]()
    partitionIndex.populate(it1.map { _._2 })
    it2.foreach(currentPoint => {
      val closePointsCount = partitionIndex.findClosePoints(currentPoint._2).size
      addPointCount(counts, currentPoint._1, closePointsCount)
    })
    counts.iterator
  }

  def findPointsCloseToBoxBounds[U <: RDD[(PointSortKey, Point)]](data: U, boxes: Iterable[MultiModalBox]): RDD[Point] = {
    val (broadcastBoxes, eps) = (data.sparkContext.broadcast(boxes), settings.epsilon)
    data.mapPartitionsWithIndex((index, it) => {
      val box = broadcastBoxes.value.find { _.partitionId == index }.get
      it.filter { p => p._2.isPointCloseToAnyBound(box, eps) }.map { _._2 }
    })
  }

  private def countClosePointsInDifferentBoxes(data: RDD[Point], boxesWithAdjacentBoxes: Iterable[MultiModalBox]): RDD[(PointSortKey, Long)] = {
    val (dist, eps) = (distance, settings.epsilon)
    val pointsInAdjacentBoxes = spatial.PointsInAdjacentBoxesRDD(data, boxesWithAdjacentBoxes)
    pointsInAdjacentBoxes.mapPartitionsWithIndex((_, it) => {
      val pointsInPartition = it.map { _._2 }.toArray.sortBy(_.distanceFromOrigin)
      val counts = mutable.HashMap[PointSortKey, Long]()
      for (i <- 1 until pointsInPartition.length) {
        var j = i - 1
        val pi = pointsInPartition(i)
        val piSortKey = new PointSortKey(pi)
        while (j >= 0 && pi.distanceFromOrigin - pointsInPartition(j).distanceFromOrigin <= eps) {
          val pj = pointsInPartition(j)
          if (pi.boxId != pj.boxId && dist(pi.obj, pj.obj) <= eps) {
            addPointCount(counts, piSortKey, 1)
            addPointCount(counts, new PointSortKey(pj), 1)
          }
          j -= 1
        }
      }
      counts.iterator
    })
  }

  private def addPointCount(counts: mutable.HashMap[PointSortKey, Long], sortKey: PointSortKey, c: Long): Option[Long] =
    if (counts.contains(sortKey)) counts.put(sortKey, counts(sortKey) + c) else counts.put(sortKey, c)
}
