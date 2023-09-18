package ru.ifmo.rain.algorithms.dbscan.spatial

import ru.ifmo.rain.algorithms.dbscan.box.{BoxCalculator, BoxTreeItemWithPoints, DimBound, MultiModalBox}
import ru.ifmo.rain.algorithms.dbscan.point.Point
import ru.ifmo.rain.algorithms.dbscan.{Settings, box}
import ru.ifmo.rain.distances.MultiDistance
import ru.ifmo.rain.utils.Compares.deq

import java.lang.Math.{abs, max, min}
import scala.annotation.tailrec
import scala.collection.mutable.{ArrayBuffer, ListBuffer}


class PartitionIndex(val partitionBounds: MultiModalBox)(implicit settings: Settings, dist: MultiDistance) extends Serializable {
  private val boxesTree = PartitionIndex.buildTree(partitionBounds)
  private val largeBox = PartitionIndex.createBoxTwiceLargerThanLeaf(boxesTree)

  def populate[T <: Iterable[Point]](points: T): Unit = populate(points.iterator)

  def populate(points: Array[Point]): Unit = populate(points.iterator)

  def populate(points: Iterator[Point]): Unit = points.foreach { pt => findBoxAndAddPoint(pt, boxesTree) }

  def findClosePoints(pt: Point): Iterable[Point] = findPotentiallyClosePoints(pt)
    .filter { p => p.pointId != pt.pointId && dist(p.obj, pt.obj) <= settings.epsilon }

  private def findPotentiallyClosePoints(pt: Point): Iterable[Point] = {
    val (box1, result) = (findBoxForPoint(pt, boxesTree), ListBuffer[Point]())
    result ++= box1.points.filter { p =>
      p.pointId != pt.pointId && abs(p.distanceFromOrigin - pt.distanceFromOrigin) <= settings.epsilon
    }

    if (pt.isPointCloseToAnyBound(box1.box, settings.epsilon)) {
      box1.adjacentBoxes.foreach(box2 => {
        if (box.MultiModalBox(pt, largeBox).isPointWithin(box2.box.centerPoint)) {
          result ++= box2.points.filter(p => abs(p.distanceFromOrigin - pt.distanceFromOrigin) <= settings.epsilon)
        }
      })
    }
    result
  }

  private def findBoxAndAddPoint(pt: Point, root: BoxTreeItemWithPoints): Unit = findBoxForPoint(pt, root).points += pt

  @tailrec
  private def findBoxForPoint(pt: Point, root: BoxTreeItemWithPoints): BoxTreeItemWithPoints = {
    if (root.children.isEmpty) root
    else root.children.find { _.box.isPointWithin(pt) } match {
      case b: Some[BoxTreeItemWithPoints] => findBoxForPoint(pt, b.get)
      case _ => throw new Exception(s"Box for point $pt was not found")
    }
  }
}


object PartitionIndex {
  def buildTree(boundingBox: MultiModalBox)(implicit settings: Settings, dist: MultiDistance): BoxTreeItemWithPoints = {
    val sortedBoxes = BoxCalculator.splitBoxIntoEqualBoxes(boundingBox).toArray.sortWith(_ < _)
    buildTree(boundingBox, sortedBoxes)
  }

  def buildTree(boundingBox: MultiModalBox, sortedBoxes: Array[MultiModalBox])
               (implicit dist: MultiDistance): BoxTreeItemWithPoints = {
    val leafs = sortedBoxes.map { new BoxTreeItemWithPoints(_) }
    leafs.foreach { leaf => leaf.adjacentBoxes ++= findAdjacentBoxes(leaf, leafs) }
    val root = new BoxTreeItemWithPoints(boundingBox)
    root.children = generateSubItems(root, 0, leafs, 0, leafs.length - 1).toList
    root
  }

  private def generateSubItems(root: BoxTreeItemWithPoints, boundDim: Int, leafs: Array[BoxTreeItemWithPoints], start: Int, end: Int)
                              (implicit dist: MultiDistance): Iterable[BoxTreeItemWithPoints] = {
    var result: List[BoxTreeItemWithPoints] = Nil
    if (boundDim < root.box.globalBounds.length) {
      var nodeStart = start
      var nodeEnd = start

      while (nodeStart <= end) {
        val bound = leafs(nodeStart).box.globalBounds(boundDim)
        val leafsSubset = ArrayBuffer[BoxTreeItemWithPoints]()
        while (nodeEnd <= end && leafs(nodeEnd).box.globalBounds(boundDim) == bound) {
          leafsSubset += leafs(nodeEnd)
          nodeEnd += 1
        }

        var newSubItem: BoxTreeItemWithPoints = null
        if (leafsSubset.size > 1) {
          val embracingBox = PartitionIndex.generateEmbracingBox(leafsSubset, root.box.globalBounds.length)
          newSubItem = new BoxTreeItemWithPoints(embracingBox)
          newSubItem.children = generateSubItems(newSubItem, boundDim + 1, leafs, nodeStart, nodeEnd - 1).toList
        }
        else if (leafsSubset.size == 1) {
          newSubItem = leafsSubset(0)
        }

        result = newSubItem :: result
        nodeStart = nodeEnd
      }
    }

    result.reverse
  }

  private def generateEmbracingBox(subItems: Iterable[BoxTreeItemWithPoints], dims: Int)(implicit dist: MultiDistance): MultiModalBox = {
    val dimensions: ArrayBuffer[DimBound] = ArrayBuffer[DimBound]()
    (0 until dims).foreach { i =>
      val zeroValue = new DimBound(Double.MaxValue, Double.MinValue, false)
      val newDimension = subItems.map(_.box.globalBounds(i)).fold(zeroValue) { (a, b) =>
        new DimBound(min(a.lower, b.lower), max(a.upper, b.upper), a.inclusive || b.inclusive)
      }
      dimensions += newDimension
    }
    new MultiModalBox(dimensions.toArray)
  }

  private def findAdjacentBoxes(x: BoxTreeItemWithPoints, boxes: Iterable[BoxTreeItemWithPoints])
  : Iterable[BoxTreeItemWithPoints] = boxes.filter { y => y != x && x.box.isAdjacentToBox(y.box) }

  private def createBoxTwiceLargerThanLeaf(root: BoxTreeItemWithPoints): MultiModalBox = {
    val leaf = findFirstLeafBox(root)
    leaf.extendBySizeOfOtherBox(leaf)
  }

  private def findFirstLeafBox(root: BoxTreeItemWithPoints): MultiModalBox = {
    var result = root
    while (result.children.nonEmpty) {
      result = result.children.head
    }
    result.box
  }
}
