package ru.ifmo.rain.algorithms.meanshift

import org.apache.spark.sql.{DataFrame, Row}
import ru.ifmo.rain.algorithms.ClusteringAlgo
import ru.ifmo.rain.algorithms.ClusteringAlgo.checkClustersAmount
import ru.ifmo.rain.distances.MultiDistance
import ru.ifmo.rain.{Sparkling, withTime}

import scala.collection.mutable.ListBuffer


/**
 * Mean Shift is an iterative clustering algorithm for machine learning and computer vision that shifts cluster
 * centroids towards the mean of local data points until convergence. It defines clusters by their mode and can handle
 * datasets with non-uniform density or complex shapes.
 * @param radius mean-object's neighbourhood radius
 * @param maxClusters upper bound for number of clusters
 * @param maxIterations maximum number of steps after which the stop condition is met
 * @param initial number of initial candidates. Should fit into driver memory
 * @param convergence the minimum distance considered significant.
 *                    If means shifted less than param, then stop condition is met
 * @param seed random seed for picking initial candidates
 */
@Sparkling
class MeanShift(
                 val radius: Double,
                 val maxClusters: Int,
                 val maxIterations: Int,
                 val initial: Int,
                 val convergence: Double = 1e-7,
                 val seed: Long = 42L
               ) extends ClusteringAlgo[MeanShiftModel] {

  override def verifyParams: Option[String] = {
    if (radius <= 0.0) Option("radius")
    else if (maxClusters < 2) Option("maxClusters")
    else if (maxIterations < 1) Option("maxIterations")
    else if (initial < 2) Option("initial")
    else if (convergence < 0.0) Option("convergence")
    else Option.empty
  }

  @Sparkling
  override def fit(df: DataFrame, dist: MultiDistance): MeanShiftModel = withTime("MeanShift", clear = true) {
    var candidates = df.rdd.takeSample(withReplacement = false, initial, seed).map(_ -> 1L)
    val converged = ListBuffer[(Row, Long)]()
    var it = 0
    while (it < maxIterations && !candidates.isEmpty) {
      val sumShifts = calcSumShifts(df, dist, candidates)
      val newCandidates = ListBuffer[(Row, Long)]()
      sumShifts.foreach { case (j, (sumShift, amount)) =>
        val newMean = dist.scale(sumShift, 1.0 / amount)
        if (dist(candidates(j)._1, newMean) > convergence) {
          newCandidates += (newMean -> amount)
        } else {
          converged += (newMean -> amount)
        }
      }
      candidates = newCandidates.toArray
      it += 1
    }
    val means = if (it == maxIterations) converged ++ candidates else converged
    val uniqueMeans = calcUniqueMeans(means.toArray, dist)
    checkClustersAmount(uniqueMeans.length, maxClusters)
    new MeanShiftModel(uniqueMeans, dist, radius)
  }

  private def calcSumShifts(df: DataFrame, dist: MultiDistance, candidates: Array[(Row, Long)]): collection.Map[Int, (Row, Long)] = {
    val (sc, r) = (df.sparkSession.sparkContext, radius)
    val broadcastMeans = sc.broadcast(candidates)

    df.rdd.mapPartitions { objects =>
      val curMeans = broadcastMeans.value
      val shifts = Array.fill(curMeans.length)(dist.zeroObj)
      val amounts = Array.fill(curMeans.length)(0L)

      objects.foreach { obj =>
        for (idx <- curMeans.indices) {
          if (dist(curMeans(idx)._1, obj) <= r) {
            amounts(idx) += 1L
            shifts(idx) = dist.plus(shifts(idx), obj)
          }
        }
      }

      Iterator.tabulate(shifts.length) { j => (j, (shifts(j), amounts(j))) }
        .filter { case (_, (_, amount)) => amount > 0L }

    }.reduceByKey { case ((shift1, amount1), (shift2, amount2)) =>
      (dist.plus(shift1, shift2), amount1 + amount2)
    }.collectAsMap()
  }

  private def calcUniqueMeans(candidates: Array[(Row, Long)], dist: MultiDistance): Array[Row] = {
    val means = candidates
      .sortBy(_._2)(Ordering.Long.reverse)
      .map(_._1)
    val unique = Array.fill[Boolean](means.length)(true)
    val uniqueMeans = ListBuffer.empty[Row]
    for (idx <- means.indices) {
      if (unique(idx)) {
        uniqueMeans += means(idx)
        for (other <- idx + 1 until means.length)
          unique(other) &&= dist(means(idx), means(other)) >= radius
      }
    }
    uniqueMeans.toArray
  }
}
