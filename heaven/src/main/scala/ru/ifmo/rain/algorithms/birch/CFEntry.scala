package ru.ifmo.rain.algorithms.birch

import org.apache.spark.sql.Row
import ru.ifmo.rain.distances.MultiDistance

class CFEntry(var n: Int, var ls: Row, var child: CFNode) extends Serializable {
  def update(that: CFEntry)(implicit dist: MultiDistance): Unit = {
    ls = dist.plus(ls, that.ls)
    n += that.n
  }

  def canMerge(that: CFEntry, distThreshold: Double)(implicit dist: MultiDistance): Boolean =
    dist(centroid, that.centroid) <= distThreshold

  def centroid(implicit dist: MultiDistance): Row = dist.scale(ls, 1.0 / n)

  def distTo(other: CFEntry)(implicit dist: MultiDistance): Double = dist(this.centroid, other.centroid)
}

object CFEntry {
  def apply(obj: Row) = new CFEntry(1, obj, null)
}