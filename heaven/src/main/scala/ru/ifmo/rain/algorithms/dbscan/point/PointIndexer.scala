package ru.ifmo.rain.algorithms.dbscan.point

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import ru.ifmo.rain.algorithms.dbscan.box.MultiModalBox
import ru.ifmo.rain.distances.MultiDistance

import java.lang.Math.{floor, pow, round}
import scala.math.log10


class PointIndexer(val numberOfPartitions: Int, val currentPartition: Int) {
  private val multiplier: Long = round(pow(10, floor(log10(numberOfPartitions)) + 1))
  private var currentIndex = 0

  def getNextIndex: Long = {
    currentIndex += 1
    currentIndex * multiplier + currentPartition
  }
}


object PointIndexer {
  def addMetadataToPoints(data: RDD[Point], boxes: Broadcast[Iterable[MultiModalBox]])
                         (implicit dist: MultiDistance): RDD[(PointSortKey, Point)] = {
    val numPartitions = data.partitions.length
    val origin = new Point(dist.zeroObj)

    data.mapPartitionsWithIndex((partitionIndex, points) => {
      val pointIndexer = new PointIndexer(numPartitions, partitionIndex)
      points.map (pt => {
        val pointIndex = pointIndexer.getNextIndex
        val box = boxes.value.find { _.isPointWithin(pt) }
        val originDist = dist(pt.obj, origin.obj)
        val boxId = box match {
          case existingBox: Some[MultiModalBox] => existingBox.get.boxId
          case _ => 0 // throw an exception?
        }

        val newPoint = new Point(pt.obj, pointIndex, boxId, originDist, pt.neighboursCount, pt.clusterId)
        new PointSortKey(newPoint) -> newPoint
      })
    })

  }
}
