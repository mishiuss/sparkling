package ru.ifmo.rain.algorithms.dbscan.spatial

import org.apache.spark.Partitioner
import ru.ifmo.rain.algorithms.dbscan.box.MultiModalBox
import ru.ifmo.rain.algorithms.dbscan.point.{Point, PointSortKey}


class BoxPartitioner(val boxes: Iterable[MultiModalBox]) extends Partitioner {
  assert { boxes.forall { _.partitionId >= 0 } }

  private val boxIdsToPartitions = generateBoxIdsToPartitionsMap(boxes)

  override def numPartitions: Int = boxes.size

  def getPartition(key: Any): Int = {
    key match {
      case k: PointSortKey => boxIdsToPartitions(k.boxId)
      case boxId: Int => boxIdsToPartitions(boxId)
      case pt: Point => boxIdsToPartitions(pt.boxId)
      case _ => throw new IllegalStateException(s"Unknown key for partition: $key")
    }
  }

  private def generateBoxIdsToPartitionsMap (boxes: Iterable[MultiModalBox]): Map[Int, Int] =
    boxes.map { x => (x.boxId, x.partitionId) }.toMap
}

object BoxPartitioner {
  def assignPartitionIdsToBoxes(boxes: Iterable[MultiModalBox]): Iterable[MultiModalBox] =
    boxes.zip(0 until boxes.size).map { x => x._1.withPartitionId(x._2) }
}