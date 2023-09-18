package ru.ifmo.rain.algorithms.cure

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK
import ru.ifmo.rain.algorithms.ClusteringAlgo
import ru.ifmo.rain.algorithms.cure.CURE._
import ru.ifmo.rain.distances.MultiDistance
import ru.ifmo.rain.{Sparkling, withTime}

/**
 * CURE (Clustering Using Representatives) is a hierarchical clustering algorithm designed for large datasets.
 * It involves representative selection, cluster assignment, local clustering, cluster merging, and shrinking steps,
 * repeated until the desired number of clusters is achieved or a stopping criterion is met.
 * @param k number of clusters
 * @param representatives number of initial data points
 * @param shrinkFactor factor that impacts moving speed of the representatives towards centroid
 * @param removeOutliers true, if algorithm should label outliers with noise label
 */
@Sparkling
class CURE(
            val k: Int,
            val representatives: Int,
            val shrinkFactor: Double,
            val removeOutliers: Boolean
          ) extends ClusteringAlgo[CUREModel] {

  override def verifyParams: Option[String] = {
    if (k < 2) Option("k")
    else if (representatives <= 0) Option("representatives")
    else if (shrinkFactor <= 0.0 || shrinkFactor >= 1.0) Option("shrinkFactor")
    else Option.empty
  }

  @Sparkling
  override def fit(df: DataFrame, dist: MultiDistance): CUREModel = withTime("CURE", clear = true) {
    val points = df.rdd.map(row => {
      val point = KDPoint(row)
      point.cluster = CURECluster(Array(point), Array(point), null, point)
      point
    }).persist(MEMORY_AND_DISK)
    val clusters = makeClusters(points, dist).collect()

    val reducedPoints = clusters.flatMap(_.representatives).toList
    val kdTree = createKDTree(reducedPoints, dist)
    val cHeap = createHeapFromClusters(clusters.toList, kdTree, dist)
    val clustersShortOfMReps =
      if (removeOutliers)
        clusters.count(_.representatives.length < representatives)
      else 0

    trimClusters(cHeap, kdTree, clustersShortOfMReps, dist)
    cHeap.getDataArray
      .slice(0, cHeap.heapSize)
      .filter(_.representatives.length >= representatives)
      .zipWithIndex
      .foreach { case (x, i) => x.id = i }

    new CUREModel(kdTree)
  }

  private def makeClusters(points: RDD[KDPoint], dist: MultiDistance): RDD[CURECluster] = {
    val (nClusters, nRepresentatives, shrink, withoutNoise) = (k, representatives, shrinkFactor, removeOutliers)
    points.mapPartitions { partition =>
      val partitionList = partition.toList

      if (partitionList.length <= nClusters)
        partitionList.map(p => CURECluster(Array(p), Array(p), null, p)).toIterator
      else {
        val kdTree = createKDTree(partitionList, dist)
        val cHeap = createHeap(partitionList, kdTree, dist)

        if (withoutNoise) {
          computeClustersAtPartitions(nClusters * 2, nRepresentatives, shrink, kdTree, cHeap, dist)
          for (i <- 0 until cHeap.heapSize)
            if (cHeap.getDataArray(i).representatives.length < nRepresentatives)
              cHeap.remove(i)
        }
        computeClustersAtPartitions(nClusters, nRepresentatives, shrink, kdTree, cHeap, dist)

        cHeap.getDataArray.slice(0, cHeap.heapSize).map { c =>
          c.points.foreach(_.cluster = null)
          val farthest = findMFarthestPoints(c.points, c.mean, nRepresentatives, dist)
          val newCluster = CURECluster(farthest, c.representatives, null, c.mean, c.distance)
          newCluster.representatives.foreach(_.cluster = newCluster)
          newCluster
        }.toIterator
      }
    }
  }

  private def trimClusters(cHeap: MinHeap, kdTree: KDTree, clustersShortOfMReps: Int, dist: MultiDistance): Unit = {
    var cl = clustersShortOfMReps
    while (cHeap.heapSize - cl > k) {
      val c1 = cHeap.takeHead()
      val nearest = c1.nearest
      val c2 = merge(c1, nearest, representatives, shrinkFactor, dist)

      if (removeOutliers) {
        val a = nearest.representatives.length < representatives
        val b = c1.representatives.length < representatives
        val c = c2.representatives.length < representatives

        if (a && b && c) cl = cl - 1
        else if (a && b) cl = cl - 2
        else if (a || b) cl = cl - 1
      }

      c1.representatives.foreach(kdTree.delete)
      nearest.representatives.foreach(kdTree.delete)

      val (newNearestCluster, nearestDistance) = getNearestCluster(c2, kdTree, dist)
      c2.nearest = newNearestCluster
      c2.distance = nearestDistance

      c2.representatives.foreach(kdTree.insert)
      removeClustersFromHeapUsingReps(kdTree, cHeap, c1, nearest, dist)
      cHeap.insert(c2)
    }
  }

  private def createHeapFromClusters(data: List[CURECluster], kdTree: KDTree, dist: MultiDistance): MinHeap = {
    val cHeap = MinHeap(data.length)
    data.foreach { p =>
      val (closest, distance) = getNearestCluster(p, kdTree, dist)
      p.nearest = closest
      p.distance = distance
      cHeap.insert(p)
    }
    cHeap
  }

  private def computeClustersAtPartitions(
                                           numClusters: Int,
                                           numRepresentatives: Int,
                                           shrink: Double,
                                           kdTree: KDTree,
                                           cHeap: MinHeap,
                                           dist: MultiDistance
                                         ): Unit = {
    while (cHeap.heapSize > numClusters) {
      val c1 = cHeap.takeHead()
      val nearest = c1.nearest
      val c2 = merge(c1, nearest, numRepresentatives, shrink, dist)

      c1.representatives.foreach(kdTree.delete)
      nearest.representatives.foreach(kdTree.delete)

      val (newNearestCluster, nearestDistance) = getNearestCluster(c2, kdTree, dist)
      c2.nearest = newNearestCluster
      c2.distance = nearestDistance
      c2.representatives.foreach(kdTree.insert)

      removeClustersFromHeapUsingReps(kdTree, cHeap, c1, nearest, dist)

      cHeap.insert(c2)
    }
  }

  private def removeClustersFromHeapUsingReps(
                                               kdTree: KDTree,
                                               cHeap: MinHeap,
                                               cluster: CURECluster,
                                               nearest: CURECluster,
                                               dist: MultiDistance
                                             ): Unit = {
    val heapSize = cHeap.heapSize
    var i = 0
    while (i < heapSize) {
      var continue = true
      val currCluster = cHeap.getDataArray(i)
      val currNearest = currCluster.nearest
      if (currCluster == nearest) {
        cHeap.remove(i)
        continue = false
      }
      if (currNearest == nearest || currNearest == cluster) {
        val (newCluster, newDistance) = getNearestCluster(currCluster, kdTree, dist)
        currCluster.nearest = newCluster
        currCluster.distance = newDistance
        cHeap.heapify(i)
        continue = false
      }
      if (continue) i += 1
    }
  }

  private def getNearestCluster(cluster: CURECluster, kdTree: KDTree, dist: MultiDistance): (CURECluster, Double) = {
    val (nearestRep, nearestDistance) = cluster
      .representatives
      .foldLeft(null: KDPoint, Double.MaxValue) {
        case ((currNearestRep, currNearestDistance), rep) =>
          val nearestRep = kdTree.closestPointOfOtherCluster(rep)
          val nearestDistance = dist(rep.obj, nearestRep.obj)
          if (nearestDistance < currNearestDistance)
            (nearestRep, nearestDistance)
          else
            (currNearestRep, currNearestDistance)
      }
    (nearestRep.cluster, nearestDistance)
  }
}

object CURE {
  private def createKDTree(data: List[KDPoint], dist: MultiDistance): KDTree = {
    val node = KDNode(data.head, null, null)
    val kdTree = KDTree(node, dist)
    for (i <- 1 until data.length)
      kdTree.insert(data(i))
    kdTree
  }

  private def createHeap(data: List[KDPoint], kdTree: KDTree, dist: MultiDistance) = {
    val cHeap = MinHeap(data.length)
    data.map { p =>
      val closest = kdTree.closestPointOfOtherCluster(p)
      p.cluster.nearest = closest.cluster
      p.cluster.distance = dist(p.obj, closest.obj)
      cHeap.insert(p.cluster)
      p.cluster
    }
    cHeap
  }

  private def merge(cluster: CURECluster, nearest: CURECluster, repr: Int, shrink: Double, dist: MultiDistance): CURECluster = {
    val mergedPoints = cluster.points ++ nearest.points
    val mean = KDPoint(dist.centroid(mergedPoints.filter(_ != null).map(_.obj)))
    var representatives = mergedPoints
    if (mergedPoints.length > repr)
      representatives = findMFarthestPoints(mergedPoints, mean, repr, dist)
    representatives = shrinkRepresentativeArray(shrink, representatives, mean, dist)

    val mergedCl = CURECluster(mergedPoints, representatives, null, mean)

    mergedCl.representatives.foreach(_.cluster = mergedCl)
    mergedCl.points.foreach(_.cluster = mergedCl)
    mergedCl.mean.cluster = mergedCl

    mergedCl
  }

  private def findMFarthestPoints(points: Array[KDPoint], mean: KDPoint, m: Int, dist: MultiDistance): Array[KDPoint] = {
    val tmpArray = new Array[KDPoint](m)
    for (i <- 0 until m) {
      var (minimal, maximal) = (0.0, 0.0)
      var maxPoint: KDPoint = null

      points.foreach(p => {
        if (!tmpArray.contains(p)) {
          if (i == 0) minimal = dist(p.obj, mean.obj)
          else {
            minimal = tmpArray.foldLeft(Double.MaxValue) { (maxd, r) => {
              if (r == null) maxd
              else {
                val d = dist(p.obj, r.obj)
                if (d < maxd) d else maxd
              }
            }
            }
          }
          if (minimal >= maximal) {
            maximal = minimal
            maxPoint = p
          }
        }
      })
      tmpArray(i) = maxPoint
    }
    tmpArray.filter(_ != null)
  }

  private def shrinkRepresentativeArray(sf: Double, repArray: Array[KDPoint], mean: KDPoint, dist: MultiDistance): Array[KDPoint] = {
    repArray.map { p =>
      if (p == null) null
      else {
        val shift = dist.plus(mean.obj, p.obj, a = sf, b = -sf)
        KDPoint(dist.plus(p.obj, shift), p.cluster)
      }
    }
  }
}