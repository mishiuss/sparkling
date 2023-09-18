package ru.ifmo.rain.algorithms

import org.apache.spark.sql.DataFrame
import ru.ifmo.rain.distances.MultiDistance


/**
 * Base class for clustering algorithms
 * @tparam Model Corresponding ClusteringModel, which will be produced by clustering algorithm
 */
abstract class ClusteringAlgo[Model <: ClusteringModel] extends Serializable {
  verifyParams match {
    case Some(message: String) => throw new IllegalArgumentException(message)
    case _ =>
  }

  def verifyParams: Option[String]

  /**
   * Launches algorithm on specified dataframe
   * @param df Multimodal dataframe
   * @param dist Multimodal metric for df
   * @return Fitted ClusteringModel
   */
  def fit(df: DataFrame, dist: MultiDistance): Model
}


object ClusteringAlgo {
  def checkClustersAmount(k: Int, maxClusters: Int): Unit = {
    if (k > maxClusters) throw new IllegalStateException(s"Too much clusters: $k with maxClusters=$maxClusters")
    else if (k < 2) throw new IllegalStateException("Failed to split data into at least two clusters")
  }
}
