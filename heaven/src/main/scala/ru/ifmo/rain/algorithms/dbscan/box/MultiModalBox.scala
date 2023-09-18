package ru.ifmo.rain.algorithms.dbscan.box

import ru.ifmo.rain.algorithms.dbscan.point.Point
import ru.ifmo.rain.distances.MultiDistance
import ru.ifmo.rain.utils.Compares
import ru.ifmo.rain.utils.Compares.{deq, dgt, dlt}

import scala.collection.mutable


class MultiModalBox(
                     val globalBounds: Array[DimBound],
                     val boxId: Int = 0,
                     val partitionId: Int = -1,
                     var adjacentBoxes: mutable.ListBuffer[MultiModalBox] = mutable.ListBuffer()
                   )(implicit dist: MultiDistance) extends Serializable with Ordered[MultiModalBox] {

  val centerPoint: Point = calculateCenter(globalBounds)

  def this(b: MultiModalBox)(implicit dist: MultiDistance) =
    this(b.globalBounds, b.boxId, b.partitionId, b.adjacentBoxes)

  def splitAlongLongestDimension(numberOfSplits: Int, idGenerator: BoxIdGenerator): Iterable[MultiModalBox] = {
    val (bound, idx) = longestDimWithIndex
    val beforeLongest =
      if (idx > 0) globalBounds.take(idx)
      else Array[DimBound]()
    val afterLongest =
      if (idx < globalBounds.length - 1) globalBounds.drop(idx + 1)
      else Array[DimBound]()

    bound.split(numberOfSplits).map { s =>
      val newBounds = (beforeLongest :+ s) ++: afterLongest
      new MultiModalBox(newBounds, idGenerator.getNextId)
    }
  }

  def isPointWithin(pt: Point): Boolean =
    dist.flatten(pt.obj).zip(globalBounds).forall { case (x, bound) => bound.isNumberWithin(x) }

  def isBigEnough(epsilon: Double): Boolean = globalBounds.forall(_.width >= 2.0 * epsilon)

  def extendBySizeOfOtherBox(box: MultiModalBox): MultiModalBox = new MultiModalBox(
    globalBounds.zip(box.globalBounds).map { bb => bb._1.extend(bb._2) }
  )

  def withPartitionId(newPartitionId: Int): MultiModalBox =
    new MultiModalBox(globalBounds, boxId, newPartitionId, adjacentBoxes)

  override def toString: String = "Box " + globalBounds.mkString(", ") + "; id = " + boxId + "; partition = " + partitionId

  private[dbscan] def longestDimWithIndex: (DimBound, Int) =
    globalBounds.zipWithIndex.maxBy(_._1.width)

  private def calculateCenter(globalBounds: Array[DimBound]): Point = {
    val center = globalBounds.map { x => x.lower + (x.upper - x.lower) / 2 }
    new Point(dist.wrap(center))
  }

  def addAdjacentBox(b: MultiModalBox): Unit =
    adjacentBoxes += b

  override def compare(that: MultiModalBox): Int =
    Compares.compare(centerPoint.obj, that.centerPoint.obj)

  def isAdjacentToBox(that: MultiModalBox): Boolean = {
    val (adjacentBounds, notAdjacentBounds) =
      globalBounds.zip(that.globalBounds).partition { case (x, y) =>
        deq(x.lower, y.lower) || deq(x.lower, y.upper) || deq(x.upper, y.upper) || deq(x.upper, y.lower)
      }

    !adjacentBounds.isEmpty && notAdjacentBounds.forall { case (x, y) =>
      (dgt(x.lower, y.lower) && dlt(x.upper, y.upper)) || (dgt(y.lower, x.lower) && dlt(y.upper, x.upper))
    }
  }
}

object MultiModalBox {
  def apply(centerPoint: Point, size: MultiModalBox)(implicit dist: MultiDistance): MultiModalBox = {
    val newBounds = dist.flatten(centerPoint.obj).map { c => new DimBound(c, c, true) }
    new MultiModalBox(newBounds).extendBySizeOfOtherBox(size)
  }
}