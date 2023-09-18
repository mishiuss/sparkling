package ru.ifmo.rain.algorithms.bisecting

import org.apache.spark.sql.Row
import ru.ifmo.rain.distances.MultiDistance


private case class ClusterSummary(size: Long, center: Row, cost: Double)

private class ClusterSummaryAggregator(val dist: MultiDistance)
  extends Serializable {
  private var n: Long = 0L
  private var sum: Row = dist.zeroObj
  private var sumSq: Double = 0.0

  def add(obj: Row): this.type = {
    n += 1L
    val norm = dist.l2(obj)
    sumSq += norm * norm
    sum = dist.plus(sum, obj)
    this
  }

  def merge(other: ClusterSummaryAggregator): this.type = {
    n += other.n
    sumSq += other.sumSq
    sum = dist.plus(sum, other.sum)
    this
  }

  def summary: ClusterSummary = {
    val center = dist.scale(sum, 1.0 / n)
    val norm = dist.l2(center)
    val cost = math.max(sumSq - n * norm * norm, 0.0)
    ClusterSummary(n, center, cost)
  }
}