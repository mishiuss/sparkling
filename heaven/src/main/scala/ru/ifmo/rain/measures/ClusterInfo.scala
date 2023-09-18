package ru.ifmo.rain.measures

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import ru.ifmo.rain.distances.MultiDistance


class ClusterInfo(val data: DataFrame, val amount: Long, val label: Int)(implicit dist: MultiDistance) {
  lazy val centroid: Row = dist.centroid(data, amount)
  lazy val distToCentroid: RDD[Double] = {
    val (distance, cent) = (dist, centroid)
    data.rdd.map(distance(_, cent))
  }
}