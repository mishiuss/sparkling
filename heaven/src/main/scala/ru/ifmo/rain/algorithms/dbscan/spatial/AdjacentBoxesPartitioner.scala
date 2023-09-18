package ru.ifmo.rain.algorithms.dbscan.spatial

import org.apache.spark.Partitioner
import ru.ifmo.rain.algorithms.dbscan.box.{MultiModalBox, BoxCalculator}


class AdjacentBoxesPartitioner(private val adjacentBoxIdPairs: Array[(Int, Int)]) extends Partitioner {
  def this (boxesWithAdjacentBoxes: Iterable[MultiModalBox]) =
    this(BoxCalculator.generateDistinctPairsOfAdjacentBoxIds(boxesWithAdjacentBoxes).toArray)

  override def numPartitions: Int = adjacentBoxIdPairs.length

  override def getPartition(key: Any): Int = {
    key match {
      case (b1: Int, b2: Int) => adjacentBoxIdPairs.indexOf((b1, b2))
      case _ => throw new IllegalStateException(s"Unknown key for partition: $key")
    }
  }
}
