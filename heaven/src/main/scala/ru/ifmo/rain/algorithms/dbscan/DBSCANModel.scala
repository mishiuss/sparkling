package ru.ifmo.rain.algorithms.dbscan

import org.apache.spark.sql.{DataFrame, Row}
import ru.ifmo.rain.algorithms.ClusteringModel
import ru.ifmo.rain.distances.MultiDistance
import ru.ifmo.rain.{LABEL_COL, NOISE_LABEL, Sparkling}


@Sparkling
class DBSCANModel(
                   private val clustered: DataFrame,
                   private val distance: MultiDistance,
                   private val epsilon: Double,
                   private val minPoints: Long,
                   private val borderNoise: Boolean
                 ) extends ClusteringModel {

  private val withoutNoise = clustered.filter { clustered.col(LABEL_COL) =!= NOISE_LABEL }

  @Sparkling
  def dataframe(): DataFrame = clustered

  @Sparkling
  override def predict(obj: Row): Int = {
    val (dist, eps, minPts, noise) = (distance, epsilon, minPoints, borderNoise)
    val neighboursByCluster = withoutNoise.rdd
      .filter { p => dist(p, obj) <= eps }
      .map { p => (p.getAs[Int](LABEL_COL), p) }
      .countByKey()
    val possibleClusters = neighboursByCluster.filter { _._2 >= minPts - 1 }
    if (possibleClusters.nonEmpty) possibleClusters.maxBy(_._2)._1
    else if (neighboursByCluster.nonEmpty && !noise) neighboursByCluster.maxBy(_._2)._1
    else NOISE_LABEL
  }

  override def predict(df: DataFrame): DataFrame = throw new NotImplementedError()
}
