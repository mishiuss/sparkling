package ru.ifmo.rain.algorithms.dbscan.box

import org.apache.spark.rdd.RDD
import ru.ifmo.rain.algorithms.dbscan.point.Point
import ru.ifmo.rain.algorithms.dbscan.spatial.BoxPartitioner
import ru.ifmo.rain.algorithms.dbscan.Settings
import ru.ifmo.rain.distances.MultiDistance


class BoxCalculator(val data: RDD[Point]) {
  def generateDensityBasedBoxes(implicit settings: Settings, dist: MultiDistance): (Iterable[MultiModalBox], MultiModalBox) = {
    val rootBox = new MultiModalBox(calculateBounds(data))
    val boxTree = BoxCalculator.generateTreeOfBoxes(rootBox)
    val broadcastBoxTree = data.sparkContext.broadcast(boxTree)

    val partialCounts: RDD[(Int, Long)] = data.mapPartitions { it =>
      val bt = broadcastBoxTree.value.clone()
      BoxCalculator.countPointsInOnePartition(bt, it)
    }

    val totalCounts = partialCounts.foldByKey(0) { _ + _ }.collectAsMap()
    val boxesWithEnoughPoints = boxTree.flattenBoxes { x => totalCounts(x.box.boxId) >= settings.pointsInBox }

    BoxCalculator.assignAdjacentBoxes(boxesWithEnoughPoints)
    BoxPartitioner.assignPartitionIdsToBoxes(boxesWithEnoughPoints) -> rootBox
  }

  private[dbscan] def calculateBounds(data: RDD[Point])(implicit dist: MultiDistance): Array[DimBound] = {
    val (minimum, maximum) = data.map { p => (p.obj, p.obj) }.reduce { (acc, cur) =>
      (dist.agg(acc._1, cur._1, math.min), dist.agg(acc._2, cur._2, math.max))
    }
    val (globalMin, globalMax) = (dist.flatten(minimum), dist.flatten(maximum))
    globalMin.zip(globalMax).map { case (minVal, maxVal) => new DimBound(minVal, maxVal, inclusive = true) }
  }
}

object BoxCalculator {
  private def generateTreeOfBoxes(root: MultiModalBox)(implicit settings: Settings): BoxTreeItemWithCount =
    generateTreeOfBoxes(root, settings.levels, new BoxIdGenerator(root.boxId))(settings.epsilon, settings.axisSplits)

  private def generateTreeOfBoxes(root: MultiModalBox, levels: Int, idGenerator: BoxIdGenerator)
                                 (implicit eps: Double, axisSplits: Int): BoxTreeItemWithCount = {
    val result = new BoxTreeItemWithCount(root)
    result.children = if (levels > 0L)
      root
        .splitAlongLongestDimension(axisSplits, idGenerator)
        .filter { _.isBigEnough(eps) }
        .map { generateTreeOfBoxes(_, levels - 1, idGenerator) }
        .toList
    else List[BoxTreeItemWithCount]()
    result
  }

  private def countOnePoint(pt: Point, root: BoxTreeItemWithCount): Unit =
    if (root.box.isPointWithin(pt)) {
      root.numberOfPoints += 1
      root.children.foreach { countOnePoint(pt, _) }
    }

  private def countPointsInOnePartition(root: BoxTreeItemWithCount, it: Iterator[Point]): Iterator[(Int, Long)] = {
    it.foreach { pt => countOnePoint(pt, root) }
    root.flatten.map { x: BoxTreeItemWithCount => (x.box.boxId, x.numberOfPoints) }.iterator
  }

  private type DimSplits = List[List[DimBound]]

  private def generateCombinationsOfSplits(splits: DimSplits, globalIndex: Int): List[List[DimBound]] = {
    if (globalIndex < 0) List(List())
    else for {
      i <- generateCombinationsOfSplits(splits, globalIndex - 1)
      j <- splits(globalIndex)
    } yield j :: i
  }

  def splitBoxIntoEqualBoxes(rootBox: MultiModalBox)(implicit settings: Settings, dist: MultiDistance): Iterable[MultiModalBox] = {
    val splits = rootBox.globalBounds.map { _.split(settings.axisSplits, 2.0 * settings.epsilon) }
    val combinations = generateCombinationsOfSplits(splits.toList, rootBox.globalBounds.length - 1)
    for (i <- combinations.indices) yield new MultiModalBox(combinations(i).reverse.toArray, i + 1)
  }

  private[dbscan] def assignAdjacentBoxes(boxesWithEnoughPoints: Iterable[MultiModalBox]): Unit = {
    val temp = boxesWithEnoughPoints.toArray
    for (i <- temp.indices) {
      for (j <- i + 1 until temp.length) {
        if (temp(i).isAdjacentToBox(temp(j))) {
          temp(i).addAdjacentBox(temp(j))
          temp(j).addAdjacentBox(temp(i))
        }
      }
    }
  }

  def generateDistinctPairsOfAdjacentBoxIds(boxesWithAdjacentBoxes: Iterable[MultiModalBox]): Iterable[(Int, Int)] =
    for (b <- boxesWithAdjacentBoxes; ab <- b.adjacentBoxes; if b.boxId < ab.boxId) yield (b.boxId, ab.boxId)
}