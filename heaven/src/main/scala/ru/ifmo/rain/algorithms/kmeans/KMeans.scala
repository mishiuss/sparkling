package ru.ifmo.rain.algorithms.kmeans

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK
import ru.ifmo.rain.algorithms.ClusteringAlgo
import ru.ifmo.rain.distances.MultiDistance
import ru.ifmo.rain.{Sparkling, withTime}

import scala.collection.mutable.ArrayBuffer
import scala.math.min
import scala.util.Random


/**
 * The k-means algorithm is an unsupervised machine learning technique used for clustering data points into k distinct
 * clusters based on their similarities. It iteratively assigns each data point to the nearest cluster centroid and then
 * updates the centroids by calculating the mean of the data points assigned to each cluster until convergence.
 * @param k target number of clusters
 * @param maxIterations maximum number of steps after which the stop condition is met
 * @param initSteps number of steps for initialization
 * @param convergence the minimum distance considered significant.
 *                    If centroids moved less than param, then stop condition is met
 * @param seed random seed for picking initial candidates
 */
@Sparkling
class KMeans(
              val k: Int,
              val maxIterations: Int,
              val initSteps: Int = 2,
              val convergence: Double = 1e-7,
              val seed: Long = 42L
            ) extends ClusteringAlgo[KMeansModel] {

  override def verifyParams: Option[String] = {
    if (k < 2) Option("k")
    else if (maxIterations < 1) Option("maxIterations")
    else if (initSteps < 1) Option("initSteps")
    else if (convergence < 0.0) Option("convergence")
    else Option.empty
  }

  @Sparkling
  override def fit(df: DataFrame, dist: MultiDistance): KMeansModel = withTime("KMeans", clear = true) {
    if (df.count() < k.toLong) throw new IllegalStateException(s"Not enough data for $k clusters")
    val initial = initParallel(df.rdd, dist)
    val centroids = iterate(df, dist, initial)
    new KMeansModel(centroids, dist)
  }

  def iterate(df: DataFrame, dist: MultiDistance, initial: Array[Row]): Array[Row] = {
    val sc = df.rdd.sparkContext
    val centroids = initial

    var (it, cost, converged) = (0, 0.0, false)
    while (it < maxIterations && !converged) {
      val broadcastCentroids = sc.broadcast(centroids)
      val costAccumulator = sc.doubleAccumulator

      val collected = df.rdd.mapPartitions { points =>
        val curCentroids = broadcastCentroids.value
        val shifts = Array.fill(curCentroids.length)(dist.zeroObj)
        val amounts = Array.fill(curCentroids.length)(0L)

        points.foreach { obj =>
          val (cost, id) = curCentroids.zipWithIndex.map { case (c, id) => (dist(obj, c), id) }.minBy(_._1)
          costAccumulator.add(cost)
          amounts(id) += 1L
          shifts(id) = dist.plus(shifts(id), obj)
        }

        Iterator.tabulate(shifts.length) { j => (j, (shifts(j), amounts(j))) }
          .filter { case (_, (_, amount)) => amount > 0L }
      }.reduceByKey { case ((shift1, amount1), (shift2, amount2)) =>
        (dist.plus(shift1, shift2), amount1 + amount2)
      }.collectAsMap()

      broadcastCentroids.destroy()
      converged = true
      collected.foreach { case (j, (shift, amount)) =>
        val newCentroid = dist.scale(shift, 1.0 / amount)
        if (converged && dist(newCentroid, centroids(j)) >= convergence) converged = false
        centroids(j) = newCentroid
      }
      cost = costAccumulator.value
      it += 1
    }
    centroids
  }

  private def initParallel(data: RDD[Row], dist: MultiDistance): Array[Row] = {
    val s = seed
    val sample = data.takeSample(withReplacement = false, 1, s)
    require(sample.nonEmpty, s"No samples available from data")

    val centroids = ArrayBuffer[Row]()
    var newCentroids = Array(sample.head)
    centroids ++= newCentroids
    val broadcasts = ArrayBuffer[Broadcast[_]]()

    var costs = data.map { _ => Double.PositiveInfinity }
    var step = 0
    while (step < initSteps) {
      val broadcastNewCentroids = data.context.broadcast(newCentroids)
      broadcasts += broadcastNewCentroids
      val oldCosts = costs
      costs = data.zip(oldCosts).map { case (obj, cost) =>
        val newCentroidsValue = broadcastNewCentroids.value
        if (newCentroidsValue.isEmpty) cost
        else min(newCentroidsValue.map { dist(obj, _) }.min, cost)
      }.persist(MEMORY_AND_DISK)
      val sumCosts = costs.sum()

      broadcastNewCentroids.unpersist()
      oldCosts.unpersist()

      newCentroids = data.zip(costs).mapPartitionsWithIndex { (index, pointCosts) =>
        val r = new Random(s ^ 193 + index * s)
        pointCosts.filter {
          case (_, cost) => r.nextDouble() < 2.0 * cost * k / sumCosts
        }.map(_._1)
      }.collect()
      centroids ++= newCentroids
      step += 1
    }

    costs.unpersist()
    broadcasts.foreach(_.destroy())

    if (centroids.length <= k) {
      centroids.toArray
    } else {
      val broadcastDistinct = data.context.broadcast(centroids.zipWithIndex)
      val countMap = data.map { obj =>
        broadcastDistinct.value.map {
          case (c, id) => id -> dist(c, obj)
        }.minBy(_._2)._1
      }.countByValue()

      broadcastDistinct.destroy()
      val weights = centroids.indices.map(countMap.getOrElse(_, 0L).toDouble).toArray
      val localKMeans = new KMeansLocal(k, maxIterations / 3, seed)
      localKMeans(centroids.toArray, dist, weights)
    }
  }
}
