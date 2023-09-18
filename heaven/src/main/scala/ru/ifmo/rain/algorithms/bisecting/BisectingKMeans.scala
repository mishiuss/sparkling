package ru.ifmo.rain.algorithms.bisecting


import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.storage.StorageLevel
import ru.ifmo.rain.algorithms.ClusteringAlgo
import ru.ifmo.rain.algorithms.bisecting.BisectingKMeans._
import ru.ifmo.rain.distances.MultiDistance
import ru.ifmo.rain.{Sparkling, withTime}

import scala.collection.mutable
import scala.util.Random


/**
 * The algorithm divides the objects into sub-clusters using KMeans at each step,
 * starting from the entire set of objects.
 * The process continues until the number of clusters reaches k.
 * @param k number of clusters
 * @param maxIterations maximum number of steps after which the stop condition is met
 * @param minClusterSize minimal number of objects (percentage) in cluster.
 * @param convergence the minimum distance considered significant.
 *                    If centroids moved less than param, then stop condition is met
 * @param seed random seed
 */
@Sparkling
class BisectingKMeans(
                       val k: Int,
                       val maxIterations: Int,
                       val minClusterSize: Double = 1.0,
                       val convergence: Double = 1e-7,
                       val seed: Long = 42L
                     ) extends ClusteringAlgo[BisectingKMeansModel] {

  override def verifyParams: Option[String] = {
    if (k < 2) Option("k")
    else if (maxIterations < 1) Option("maxIterations")
    else if (0.0 <= minClusterSize && minClusterSize > 1.0) Option("minClusterSize")
    else Option.empty
  }

  @Sparkling
  override def fit(df: DataFrame, dist: MultiDistance): BisectingKMeansModel = withTime("BisectingKMeans") {
    var assignments = df.rdd.map(obj => (ROOT_INDEX, obj))
    var activeClusters = summarize(assignments, dist)
    val rootSummary = activeClusters(ROOT_INDEX)
    val n = rootSummary.size

    val minSize = math.ceil(minClusterSize * n).toLong
    var inactiveClusters = mutable.Seq.empty[(Long, ClusterSummary)]
    val random = new Random(seed)
    var numLeafClustersNeeded = k - 1
    var level = 1
    var preIndices: RDD[Long] = null
    var indices: RDD[Long] = null
    while (activeClusters.nonEmpty && numLeafClustersNeeded > 0 && level < LEVEL_LIMIT) {
      var divisibleClusters = activeClusters.filter { case (_, summary) =>
        (summary.size >= minSize) && (summary.cost > 1e-12 * summary.size)
      }
      if (divisibleClusters.size > numLeafClustersNeeded) {
        divisibleClusters = divisibleClusters.toSeq.sortBy {
          case (_, summary) => -summary.size
        }.take(numLeafClustersNeeded).toMap
      }
      if (divisibleClusters.nonEmpty) {
        val divisibleIndices = divisibleClusters.keys.toSet
        var newClusterCenters = divisibleClusters.flatMap { case (index, summary) =>
          val (left, right) = splitCenter(summary.center, random, dist)
          Iterator((leftChildIndex(index), left), (rightChildIndex(index), right))
        }.map(identity)
        var newClusters: Map[Long, ClusterSummary] = null
        var newAssignments: RDD[(Long, Row)] = null
        var (it, converged) = (0, false)
        while (!converged && it < maxIterations) {
          newAssignments = updateAssignments(assignments, divisibleIndices, newClusterCenters, dist)
            .filter { case (index, _) => divisibleIndices.contains(parentIndex(index)) }
          newClusters = summarize(newAssignments, dist)
          val oldClusterCenters = newClusterCenters
          newClusterCenters = newClusters.mapValues(_.center).map(identity)
          converged = oldClusterCenters.map { case (label, c) =>
            dist(c, newClusterCenters(label))
          }.forall(_ < convergence)
          it += 1
        }

        if (preIndices != null) preIndices.unpersist()
        preIndices = indices

        indices = updateAssignments(assignments, divisibleIndices, newClusterCenters, dist)
          .keys
          .persist(StorageLevel.MEMORY_AND_DISK)
        assignments = indices.zip(df.rdd)
        inactiveClusters ++= activeClusters
        activeClusters = newClusters
        numLeafClustersNeeded -= divisibleClusters.size
      } else {
        inactiveClusters ++= activeClusters
        activeClusters = Map.empty
      }
      level += 1
    }

    if (preIndices != null) preIndices.unpersist()
    if (indices != null) indices.unpersist()

    val clusters = activeClusters ++ inactiveClusters
    val root = buildTree(clusters, dist)
    new BisectingKMeansModel(root, dist)
  }
}

object BisectingKMeans {
  private val ROOT_INDEX: Long = 1
  private val MAX_DIVISIBLE_CLUSTER_INDEX: Long = Long.MaxValue / 2
  private val LEVEL_LIMIT = math.log10(Long.MaxValue) / math.log10(2)

  private def leftChildIndex(index: Long): Long = {
    require(index <= MAX_DIVISIBLE_CLUSTER_INDEX, s"Child index out of bound: 2 * $index.")
    2 * index
  }

  private def rightChildIndex(index: Long): Long = {
    require(index <= MAX_DIVISIBLE_CLUSTER_INDEX, s"Child index out of bound: 2 * $index + 1.")
    2 * index + 1
  }

  private def parentIndex(index: Long): Long = {
    index / 2
  }

  private def summarize(assignments: RDD[(Long, Row)], dist: MultiDistance): Map[Long, ClusterSummary] = {
    assignments.aggregateByKey(new ClusterSummaryAggregator(dist))(
      seqOp = (agg, v) => agg.add(v),
      combOp = (agg1, agg2) => agg1.merge(agg2)
    ).mapValues(_.summary)
      .collect().toMap
  }

  private def splitCenter(center: Row, random: Random, dist: MultiDistance): (Row, Row) = {
    val level = 1e-4 * dist.l2(center)
    val noise = dist.generate { modal =>
      val array = Array.fill(modal.dim)(random.nextDouble())
      Vectors.dense(array)
    }
    val left = dist.plus(center, noise, b = -level)
    val right = dist.plus(center, noise, b = level)
    (left, right)
  }

  private def updateAssignments(
                                 assignments: RDD[(Long, Row)],
                                 divisibleIndices: Set[Long],
                                 newClusterCenters: Map[Long, Row],
                                 dist: MultiDistance): RDD[(Long, Row)] = {
    assignments.map { case (index, v) =>
      if (divisibleIndices.contains(index)) {
        val children = Seq(leftChildIndex(index), rightChildIndex(index))
        val newClusterChildren = children.filter(newClusterCenters.contains)
        val newClusterChildrenCenterToId = newClusterChildren.map(id => newClusterCenters(id) -> id).toMap
        val newClusterChildrenCenters = newClusterChildrenCenterToId.keys.toArray
        if (newClusterChildren.nonEmpty) {
          val center = newClusterChildrenCenters.map {
            child => child -> dist(v, child)
          }.minBy(_._2)._1
          (newClusterChildrenCenterToId(center), v)
        } else {
          (index, v)
        }
      } else {
        (index, v)
      }
    }
  }

  private def buildTree(clusters: Map[Long, ClusterSummary], dist: MultiDistance): ClusteringTreeNode = {
    var leafIndex = 0
    var internalIndex = -1

    def buildSubTree(rawIndex: Long): ClusteringTreeNode = {
      val cluster = clusters(rawIndex)
      val size = cluster.size
      val center = cluster.center
      val cost = cluster.cost
      val isInternal = clusters.contains(leftChildIndex(rawIndex))
      if (isInternal) {
        val index = internalIndex
        internalIndex -= 1
        val leftIndex = leftChildIndex(rawIndex)
        val rightIndex = rightChildIndex(rawIndex)
        val indexes = Seq(leftIndex, rightIndex).filter(clusters.contains)
        val height = indexes.map { childIndex => dist(center, clusters(childIndex).center) }.max
        val children = indexes.map(buildSubTree).toArray
        new ClusteringTreeNode(index, size, center, cost, height, children)
      } else {
        val index = leafIndex
        leafIndex += 1
        val height = 0.0
        new ClusteringTreeNode(index, size, center, cost, height, Array.empty)
      }
    }

    buildSubTree(ROOT_INDEX)
  }
}
