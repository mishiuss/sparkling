package ru.ifmo.rain.algorithms.dbscan.point

import org.apache.spark.sql.Row
import ru.ifmo.rain.algorithms.dbscan._
import ru.ifmo.rain.algorithms.dbscan.box.{DimBound, MultiModalBox}
import ru.ifmo.rain.distances.MultiDistance

import scala.math.abs


class Point(
             val obj: Row,
             val pointId: Long = 0,
             val boxId: Int = 0,
             val distanceFromOrigin: Double = 0.0,
             val neighboursCount: Long = 0,
             val clusterId: Long = Int.MinValue
           ) extends Serializable {

  def this(pt: Point) = this(pt.obj, pt.pointId, pt.boxId, pt.distanceFromOrigin, pt.neighboursCount, pt.clusterId)

  def withNumberOfNeighbors(newNumber: Long): Point =
    new Point(obj, pointId, boxId, distanceFromOrigin, newNumber, clusterId)

  def withClusterId(newId: Long): Point =
    new Point(obj, pointId, boxId, distanceFromOrigin, neighboursCount, newId)

  def isPointCloseToAnyBound(box: MultiModalBox, threshold: Double)(implicit dist: MultiDistance): Boolean =
    box.globalBounds.zipWithIndex.exists { case (bound, index) =>
      dist.approxDistByGlobal(obj, bound.lower, index) <= threshold ||
        dist.approxDistByGlobal(obj, bound.upper, index) <= threshold
    }

  override def equals(that: Any): Boolean = {
    that match {
      case point: Point => point.canEqual(this) && this.obj == point.obj
      case _ => false
    }
  }

  override def hashCode(): Int = obj.hashCode()

  override def toString: String = {
    "Point at (" + obj.mkString(", ") + "); id = " + pointId + "; box = " + boxId +
      "; cluster = " + clusterId + "; neighbors = " + neighboursCount
  }

  def canEqual(other: Any): Boolean = other.isInstanceOf[Point]
}
