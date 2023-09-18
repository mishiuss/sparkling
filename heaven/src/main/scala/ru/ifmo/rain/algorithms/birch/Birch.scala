package ru.ifmo.rain.algorithms.birch

import org.apache.spark.sql.DataFrame
import ru.ifmo.rain.algorithms.ClusteringAlgo
import ru.ifmo.rain.algorithms.kmeans.KMeans
import ru.ifmo.rain.distances.MultiDistance
import ru.ifmo.rain.{Sparkling, withTime}


/**
 * Algorithm creates CF-tree and pass its leaves to KMeans algorithm
 * @param k target number of clusters
 * @param maxBranches maximum number of sub-clusters in each node
 * @param threshold radius of the sub-cluster after merging a new sample
 *                  and the closest sub-cluster must be lesser than this value
 * @param maxIterations maximum number of steps after which the stop condition is met
 */
@Sparkling
class Birch(
             val k: Int,
             val maxBranches: Int,
             val threshold: Double,
             val maxIterations: Int
           ) extends ClusteringAlgo[BirchModel] {

  override def verifyParams: Option[String] = {
    if (maxBranches < 2) Option("maxBranches")
    else if (threshold < 0.0) Option("threshold")
    else if (k < 2) Option("k")
    else if (maxIterations < 1) Option("maxIterations")
    else Option.empty
  }

  @Sparkling
  override def fit(df: DataFrame, dist: MultiDistance): BirchModel = withTime("Birch", clear = true) {
    val leafEntryCentroids = df.rdd.mapPartitions { objects =>
      val cfTree = new CFTree(maxBranches, threshold, dist).enableAutoRebuild()
      objects.foreach(obj => cfTree.insertEntry(CFEntry(obj)))
      cfTree.iterator
    }.map(entry => entry.centroid(dist))

    val numLeafEntries = leafEntryCentroids.persist().count()

    val approx = if (numLeafEntries > k)
      new KMeans(k, maxIterations).fit(df, dist).centroids()
    else
      leafEntryCentroids.collect()

    val centroids = new KMeans(k, maxIterations).iterate(df, dist, approx)
    new BirchModel(centroids, dist)
  }
}