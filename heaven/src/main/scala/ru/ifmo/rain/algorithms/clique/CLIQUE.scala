package ru.ifmo.rain.algorithms.clique

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.Vector
import ru.ifmo.rain.algorithms.ClusteringAlgo
import ru.ifmo.rain.algorithms.clique.CLIQUEModel.{DenseUnit, DenseUnits}
import ru.ifmo.rain.distances.MultiDistance
import ru.ifmo.rain.{Sparkling, withTime}

import scala.collection.mutable
import scala.math.abs


/**
 * CLIQUE is a categorical data clustering algorithm that identifies cohesive groups of data items sharing common
 * attribute values. It works in two steps: initially forming cliques by iteratively merging similar ones, and then
 * pruning cliques to remove outliers based on cohesion and density. The algorithm considers both intra-clique
 * similarity and inter-clique dissimilarity to identify significant clusters.
 * @param threshold ratio of total number of points
 * @param splits number of subdivisions for a single dimension
 * @param levels number of attempts to cartesian grids
 */
@Sparkling
class CLIQUE(val threshold: Double, val splits: Int, val levels: Int) extends ClusteringAlgo[CLIQUEModel] {
  override def verifyParams: Option[String] = {
    if (threshold <= 0.0 || threshold >= 1.0) Option("threshold")
    else if (splits < 2) Option("splits")
    else if (levels < 1) Option("levels")
    else Option.empty
  }

  @Sparkling
  override def fit(df: DataFrame, dist: MultiDistance): CLIQUEModel = withTime("CLIQUE") {
    implicit val tau: Long = (df.count() * threshold).toLong

    var units = initialGrid(df, dist)
    if (units.isEmpty) throw new IllegalStateException("Failed to get one-dimensional dense units")

    for (level <- 2 to levels) {
      units = multiDimensionalGrid(df, units, level, dist)
      if (units.isEmpty) throw new IllegalStateException(s"Failed to get $level-dimensional units")
    }

    val clustersUnits = clusterUnits(units)
    new CLIQUEModel(clustersUnits, splits, dist)
  }

  private def initialGrid(df: DataFrame, dist: MultiDistance)(implicit tau: Long): DenseUnits = {
    val s = splits
    val buckets = df.rdd.flatMap { row =>
      dist.flatten(row).zipWithIndex.map { case (value, globalIdx) =>
        val bucket = value * s % s
        (globalIdx, bucket.floor.toInt)
      }
    }

    buckets.countByValue()
      .filter { case (_, count) => count >= tau }
      .map { case ((globalIdx, bucket), _) => Map(globalIdx -> bucket) }
      .toArray
  }

  private def clusterUnits(units: DenseUnits): IndexedSeq[DenseUnits] = {
    val edges = buildDenseUnitsGraph(units)
    val (num, components) = connectedComponents(edges)
    val idxComponents = components.zipWithIndex
    for (c <- 0 until num) yield {
      idxComponents
        .filter { idxComp => idxComp._1 == c }
        .map { idxComp => units(idxComp._2) }
    }
  }

  private def multiDimensionalGrid(df: DataFrame, units: DenseUnits, level: Int, dist: MultiDistance)
                                  (implicit tau: Long): DenseUnits = {
    val candidates = selfJoinWithPrune(units, level)
    val (s, sc) = (splits, df.sparkSession.sparkContext)
    val idxCandidates = sc.broadcast(candidates.zipWithIndex)

    val projectionCounts = df.rdd.flatMap { row =>
      idxCandidates.value.filter { case (candidate, _) =>
        candidate.forall { case (globalIdx, bucket) =>
          val value = dist.byGlobal(row, globalIdx)
          (value * s % s).floor.toInt == bucket
        }
      }.map(_._2)
    }.countByValue()

    projectionCounts
      .filter { case (_, count) => count >= tau }
      .map { case (id, _) => candidates(id) }
      .toArray
  }

  private def selfJoinWithPrune(units: DenseUnits, level: Int): DenseUnits = {
    val candidates = mutable.Set[DenseUnit]()
    val prevUnitsAsSet = units.toSet
    for (x <- units.indices; y <- x + 1 until units.length) {
      val joined = units(x) ++ units(y)
      if (joined.size == level && subLevelsIncluded(joined, prevUnitsAsSet)) {
        candidates += joined
      }
    }
    candidates.toArray
  }

  private def subLevelsIncluded(candidate: DenseUnit, prevUnits: Set[DenseUnit]): Boolean =
    candidate.forall { case (level, _) => prevUnits.contains(candidate - level) }

  private def connectedComponents(implicit edges: IndexedSeq[Array[Int]]): (Int, Array[Int]) = {
    implicit val components: Array[Int] = Array.fill(edges.size)(-1)
    var componentId = 0
    for (idx <- components.indices if components(idx) == -1) {
      dfs(idx, componentId)
      componentId += 1
    }
    (componentId, components)
  }

  private def dfs(idx: Int, componentId: Int)
                 (implicit edges: IndexedSeq[Array[Int]], components: Array[Int]): Unit = {
    components(idx) = componentId
    for (next <- edges(idx) if components(next) == -1) dfs(next, componentId)
  }

  private def buildDenseUnitsGraph(units: DenseUnits): IndexedSeq[Array[Int]] = {
    for (xIdx <- units.indices) yield {
      units.indices.filter { yIdx =>
        val (x, y) = (units(xIdx), units(yIdx))
        if (x.keySet != y.keySet) false
        else x.map { case (key, bucket) => abs(bucket - y(key)) }.sum <= 1
      }.toArray
    }
  }
}
