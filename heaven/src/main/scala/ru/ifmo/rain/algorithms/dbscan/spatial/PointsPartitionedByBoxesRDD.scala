package ru.ifmo.rain.algorithms.dbscan.spatial

import org.apache.spark.rdd.{RDD, ShuffledRDD}
import org.apache.spark.sql.Row
import ru.ifmo.rain.algorithms.dbscan._
import ru.ifmo.rain.algorithms.dbscan.box.{MultiModalBox, BoxCalculator}
import ru.ifmo.rain.algorithms.dbscan.point.{Point, PointIndexer, PointSortKey}
import ru.ifmo.rain.distances.MultiDistance


class PointsPartitionedByBoxesRDD(prev: RDD[(PointSortKey, Point)], val boxes: Iterable[MultiModalBox], val boundingBox: MultiModalBox)
  extends ShuffledRDD[PointSortKey, Point, Point](prev, new BoxPartitioner(boxes))

object PointsPartitionedByBoxesRDD {
  def apply(rawData: RDD[Point])(implicit settings: Settings, dist: MultiDistance): PointsPartitionedByBoxesRDD = {
    val (boxes, boundingBox) = new BoxCalculator(rawData).generateDensityBasedBoxes
    val pointsInBoxes = PointIndexer.addMetadataToPoints(rawData, rawData.sparkContext.broadcast(boxes))
    PointsPartitionedByBoxesRDD(pointsInBoxes, boxes, boundingBox)
  }

  def apply(pointsInBoxes: RDD[(PointSortKey, Point)], boxes: Iterable[MultiModalBox], boundingBox: MultiModalBox): PointsPartitionedByBoxesRDD =
    new PointsPartitionedByBoxesRDD(pointsInBoxes, boxes, boundingBox)
}


