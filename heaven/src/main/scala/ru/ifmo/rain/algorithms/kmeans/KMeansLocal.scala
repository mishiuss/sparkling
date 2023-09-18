package ru.ifmo.rain.algorithms.kmeans

import org.apache.spark.sql.Row
import ru.ifmo.rain.utils.Compares.deq
import ru.ifmo.rain.distances.MultiDistance

import scala.math.min
import scala.util.Random


class KMeansLocal(val k: Int, maxIterations: Int, seed: Long) {
  def apply(data: Array[Row], dist: MultiDistance, weights: Array[Double]): Array[Row] = {
    val (centroids, random) = (new Array[Row](k), new Random(seed))
    centroids(0) = pickWeighted(random, data, weights)
    val costArray = data.map {
      dist(_, centroids(0))
    }

    for (i <- 1 until k) {
      centroids(i) = pickWeighted(random, data, costArray.zip(weights).map(p => p._1 * p._2))
      for (p <- data.indices) costArray(p) = min(dist(data(p), centroids(i)), costArray(p))
    }

    val oldClosest = Array.fill(data.length)(-1)
    val zipCentroids = centroids.zipWithIndex
    val zipData = data.zipWithIndex

    var (it, moved) = (0, true)
    while (moved && it < maxIterations) {
      moved = false
      val counts = Array.ofDim[Double](k)
      val objAccumulator = Array.fill(k)(dist.zeroObj)
      for ((obj, i) <- zipData) {
        val idx = zipCentroids.map { case (c, cIdx) => cIdx -> dist(c, obj) }.minBy(_._2)._1
        objAccumulator(idx) = dist.plus(obj, objAccumulator(idx), a = weights(i))
        counts(idx) += weights(i)
        if (idx != oldClosest(i)) {
          moved = true
          oldClosest(i) = idx
        }
      }

      for (j <- 0 until k) {
        if (deq(counts(j), 0.0)) centroids(j) = data(random.nextInt(data.length))
        else centroids(j) = dist.scale(objAccumulator(j), 1.0 / counts(j))
      }
      it += 1
    }
    centroids
  }

  private def pickWeighted(rand: Random, data: Array[Row], weights: Array[Double]): Row = {
    val r = rand.nextDouble() * weights.sum
    var (i, curWeight) = (0, 0.0)
    while (i < data.length && curWeight < r) {
      curWeight += weights(i)
      i += 1
    }
    data(i - 1)
  }
}
