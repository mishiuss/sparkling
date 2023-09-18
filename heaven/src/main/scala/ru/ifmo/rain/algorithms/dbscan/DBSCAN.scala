package ru.ifmo.rain.algorithms.dbscan

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK
import ru.ifmo.rain.algorithms.ClusteringAlgo
import ru.ifmo.rain.algorithms.ClusteringAlgo.checkClustersAmount
import ru.ifmo.rain.algorithms.ClusteringModel.withLabel
import ru.ifmo.rain.algorithms.dbscan.box.MultiModalBox
import ru.ifmo.rain.algorithms.dbscan.point._
import ru.ifmo.rain.algorithms.dbscan.spatial.{PartitionIndex, PointsInAdjacentBoxesRDD, PointsPartitionedByBoxesRDD}
import ru.ifmo.rain.distances.MultiDistance
import ru.ifmo.rain.{LABEL_FIELD, NOISE_LABEL, Sparkling, withTime}

import scala.collection.immutable.HashMap
import scala.collection.mutable


/**
 * DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups points based on their distance to nearby
 * points, forming clusters of any shape and size defined by their density. DBSCAN is effective in handling datasets
 * with irregular shapes and varying density, and it requires two main parameters: epsilon (maximum distance for points
 * to be considered neighbors) and min_points (minimum points to form a dense region).
 * @param epsilon neighborhood radius
 * @param minPoints the number of objects in a point neighborhood for it to be considered as a core point
 * @param borderNoise true, if border points should be considered as noise and not be assigned to clusters
 * @param maxClusters maximum number of clusters
 * @param pointsInBox minimum number of points in a single subdivision of dimensions.
 *                    Does not affect the result of a clustering, but affects performance only
 * @param axisSplits number of splits per one dimension
 *                   Does not affect the result of a clustering, but affects performance only
 * @param levels number of procedure that split space on some dimension
 *               Does not affect the result of a clustering, but affects performance only
 */
@Sparkling
class DBSCAN(
              val epsilon: Double,
              val minPoints: Long,
              val borderNoise: Boolean,
              val maxClusters: Int,
              val pointsInBox: Long,
              val axisSplits: Int = 2,
              val levels: Int = 10
            ) extends ClusteringAlgo[DBSCANModel] {

  override def verifyParams: Option[String] = {
    if (epsilon <= 0.0) Option("epsilon")
    else if (minPoints < 2L) Option("minPoints")
    else if (maxClusters < 2) Option("maxClusters")
    else if (pointsInBox < 0L) Option("pointsInBox")
    else if (axisSplits < 1) Option("axisSplits")
    else if (levels < 1) Option("levels")
    else Option.empty
  }

  @Sparkling
  override def fit(df: DataFrame, dist: MultiDistance): DBSCANModel = withTime("DBSCAN", clear = true) {
    implicit val settings: Settings = new Settings(
      epsilon, minPoints, pointsInBox, axisSplits, levels, borderNoise
    )
    implicit val implicitDistance: MultiDistance = dist
    implicit val analyzer: DistanceAnalyzer = new DistanceAnalyzer

    val partitionedData = PointsPartitionedByBoxesRDD(df.rdd.map { new Point(_) })
    val pointsWithNeighborCounts = analyzer.countNeighborsForEachPoint(partitionedData)
    val broadcastBoxes = partitionedData.sparkContext.broadcast(partitionedData.boxes)

    val partiallyClusteredData = pointsWithNeighborCounts.mapPartitionsWithIndex((partitionIndex, it) => {
      val partitionBoundingBox = broadcastBoxes.value.find { _.partitionId == partitionIndex }.get
      findClustersInOnePartition(it, partitionBoundingBox)
    },
      preservesPartitioning = true
    ).persist(MEMORY_AND_DISK)

    val completelyClusteredData = mergeClustersFromDifferentPartitions(partiallyClusteredData, partitionedData.boxes)
    val labels = completelyClusteredData.map { _.clusterId }.filter { _ != NOISE_LABEL }.distinct()
    checkClustersAmount(labels.count().toInt, maxClusters)

    val broadcastMapping = df.sparkSession.sparkContext.broadcast(labels.collect().zipWithIndex.toMap)
    val clusteredData = completelyClusteredData.map { p =>
      withLabel(p.obj, if (p.clusterId == NOISE_LABEL) NOISE_LABEL else broadcastMapping.value(p.clusterId))
    }
    val clustered = df.sparkSession.createDataFrame(clusteredData, df.schema.add(LABEL_FIELD))
    new DBSCANModel(clustered.checkpoint(true), dist, epsilon, minPoints, borderNoise)
  }

  private type SetOfIds = mutable.HashSet[Long]

  private def findClustersInOnePartition(it: Iterator[(PointSortKey, Point)], boundingBox: MultiModalBox)
                                        (implicit settings: Settings, dist: MultiDistance): Iterator[(PointSortKey, Point)] = {
    var tempPointId = 0
    val points = it.map(x => {
      tempPointId += 1
      val newPt = new MutablePoint(x._2, tempPointId)
      (tempPointId, newPt)
    }).toMap

    val partitionIndex = new PartitionIndex(boundingBox)
    partitionIndex.populate(points.values)
    var startingPointWithId = findUnvisitedCorePoint(points)
    while (startingPointWithId.isDefined) {
      expandCluster(points, partitionIndex, startingPointWithId.get._2)
      startingPointWithId = findUnvisitedCorePoint(points)
    }
    points.map { pt => new PointSortKey(pt._2) -> pt._2.toImmutablePoint }.iterator
  }

  private def expandCluster(points: Map[Int, MutablePoint], index: PartitionIndex, startingPoint: MutablePoint)
                           (implicit settings: Settings, dist: MultiDistance): Unit = {
    val corePointsInCluster = new mutable.BitSet(points.size)
    corePointsInCluster += startingPoint.tempId
    startingPoint.transientClusterId = startingPoint.pointId
    startingPoint.visited = true

    while (corePointsInCluster.nonEmpty) {
      val currentPointId = corePointsInCluster.head
      val neighbors = findUnvisitedNeighbors(index, points(currentPointId))
      neighbors.foreach(n => {
        n.visited = true
        if (n.neighboursCount >= settings.minPoints) {
          n.transientClusterId = startingPoint.transientClusterId
          corePointsInCluster += n.tempId
        }
        else if (!settings.borderNoise) {
          n.transientClusterId = startingPoint.transientClusterId
        }
        else {
          n.transientClusterId = NOISE_LABEL
        }
      })
      corePointsInCluster -= currentPointId
    }
  }

  private def findUnvisitedNeighbors(index: PartitionIndex, pt: MutablePoint)
                                    (implicit settings: Settings, dist: MultiDistance): Iterable[MutablePoint] =
    index.findClosePoints(pt)
      .map { _.asInstanceOf[MutablePoint] }
      .filter { p => !p.visited && p.transientClusterId == Int.MinValue && dist(p.obj, pt.obj) <= settings.epsilon }

  private def findUnvisitedCorePoint(points: Map[Int, MutablePoint])
                                    (implicit settings: Settings): Option[(Int, MutablePoint)] =
    points.find { pt => !pt._2.visited && pt._2.neighboursCount >= settings.minPoints }

  private def mergeClustersFromDifferentPartitions(partiallyClusteredData: RDD[(PointSortKey, Point)], boxes: Iterable[MultiModalBox])
                                                  (implicit settings: Settings, dist: MultiDistance, analyzer: DistanceAnalyzer): RDD[Point] = {
    val pointsCloseToBoxBounds = analyzer.findPointsCloseToBoxBounds(partiallyClusteredData, boxes)
    val (mappings, borderPoints) = generateMappings(pointsCloseToBoxBounds, boxes)
    val broadcastMappings = partiallyClusteredData.sparkContext.broadcast(mappings)
    val broadcastBorderPoints = partiallyClusteredData.sparkContext.broadcast(borderPoints)

    partiallyClusteredData.mapPartitions(_.map { x =>
      reassignClusterId(x._2, broadcastMappings.value, broadcastBorderPoints.value)
    })
  }

  private def generateMappings(pointsCloseToBoxBounds: RDD[Point], boxes: Iterable[MultiModalBox])
                              (implicit settings: Settings, dist: MultiDistance): (mutable.HashSet[(SetOfIds, Long)], Map[Long, Long]) = {
    val pointsInAdjacentBoxes = PointsInAdjacentBoxesRDD(pointsCloseToBoxBounds, boxes)

    val pairwiseMappings: RDD[(Long, Long)] = pointsInAdjacentBoxes.mapPartitionsWithIndex((_, it) => {
      val pointsInPartition = it.map { _._2 }.toArray.sortBy(_.distanceFromOrigin)
      val pairs = mutable.HashSet[(Long, Long)]()

      for (i <- 1 until pointsInPartition.length) {
        var j = i - 1
        val pi = pointsInPartition(i)
        while (j >= 0 && pi.distanceFromOrigin - pointsInPartition(j).distanceFromOrigin <= settings.epsilon) {
          val pj = pointsInPartition(j)
          if (pi.boxId != pj.boxId && pi.clusterId != pj.clusterId && dist(pi.obj, pj.obj) <= settings.epsilon) {
            val enoughCorePoints = if (settings.borderNoise) isCorePoint(pi) && isCorePoint(pj)
            else isCorePoint(pi) || isCorePoint(pj)

            if (enoughCorePoints) {
              val (c1, c2) = addBorderPointToCluster(pi, pj)
              if (c1 != c2) {
                if (pi.clusterId < pj.clusterId) pairs += ((pi.clusterId, pj.clusterId))
                else pairs += ((pj.clusterId, pi.clusterId))
              }
            }
          }
          j -= 1
        }
      }
      pairs.iterator
    })

    val borderPointsToBeAssignedToClusters = if (!settings.borderNoise) {
      pointsInAdjacentBoxes.mapPartitionsWithIndex((_, it) => {
        val pointsInPartition = it.map { _._2 }.toArray.sortBy(_.distanceFromOrigin)
        val bp = scala.collection.mutable.Map[Long, Long]()
        for (i <- 1 until pointsInPartition.length) {
          var j = i - 1
          val pi = pointsInPartition(i)
          while (j >= 0 && pi.distanceFromOrigin - pointsInPartition(j).distanceFromOrigin <= settings.epsilon) {
            val pj = pointsInPartition(j)
            if (pi.boxId != pj.boxId && pi.clusterId != pj.clusterId && dist(pi.obj, pj.obj) <= settings.epsilon) {
              val enoughCorePoints = isCorePoint(pi) || isCorePoint(pj)
              if (enoughCorePoints) addBorderPointToCluster(pi, pj, bp)
            }
            j -= 1
          }
        }
        bp.iterator
      }).collect().toMap
    }
    else HashMap[Long, Long]()

    val mappings = mutable.HashSet[mutable.HashSet[Long]]()
    val processedPairs = mutable.HashSet[(Long, Long)]()
    pairwiseMappings.collect().foreach { x => processPairOfClosePoints(x._1, x._2, processedPairs, mappings) }
    val finalMappings = mappings.filter { _.nonEmpty }.map { x => (x, x.head) }
    (finalMappings, borderPointsToBeAssignedToClusters)
  }

  private def addBorderPointToCluster(pt1: Point, pt2: Point)(implicit settings: Settings): (Long, Long) = {
    var (newClusterId1, newClusterId2) = (pt1.clusterId, pt2.clusterId)
    if (!settings.borderNoise) {
      if (!isCorePoint(pt1) && pt1.clusterId == Int.MinValue) newClusterId1 = pt2.clusterId
      else if (!isCorePoint(pt2) && pt2.clusterId == Int.MinValue) newClusterId2 = pt1.clusterId
    }
    (newClusterId1, newClusterId2)
  }

  private def processPairOfClosePoints(
                                        c1: Long, c2: Long,
                                        processedPairs: mutable.HashSet[(Long, Long)],
                                        mappings: mutable.HashSet[SetOfIds]
                                      ): Unit = {

    val pair = if (c1 < c2) (c1, c2) else (c2, c1)
    if (!processedPairs.contains(pair)) {
      processedPairs += pair
      val m1 = mappings.find { _.contains(c1) }
      val m2 = mappings.find { _.contains(c2) }
      (m1, m2) match {
        case (None, None) => mappings += mutable.HashSet(c1, c2)
        case (None, y: Some[SetOfIds]) => y.get += c1
        case (x: Some[SetOfIds], None) => x.get += c2
        case (x: Some[SetOfIds], y: Some[SetOfIds]) => if (x != y) {
          mappings += x.get.union(y.get)
          x.get.clear(); y.get.clear()
        }
      }
    }
  }

  private def addBorderPointToCluster(pt1: Point, pt2: Point, borderPointsToBeAssignedToClusters: mutable.Map[Long, Long])
                                     (implicit settings: Settings): (Long, Long) = {
    var (newClusterId1, newClusterId2) = (pt1.clusterId, pt2.clusterId)
    if (!settings.borderNoise) {
      if (!isCorePoint(pt1) && pt1.clusterId == Int.MinValue) {
        borderPointsToBeAssignedToClusters.put(pt1.pointId, pt2.clusterId)
        newClusterId1 = pt2.clusterId
      }
      else if (!isCorePoint(pt2) && pt2.clusterId == Int.MinValue) {
        borderPointsToBeAssignedToClusters.put(pt2.pointId, pt1.clusterId)
        newClusterId2 = pt1.clusterId
      }
    }
    (newClusterId1, newClusterId2)
  }

  private def reassignClusterId(pt: Point, mappings: mutable.HashSet[(SetOfIds, Long)], borderPoints: Map[Long, Long]): Point = {
    var newClusterId = pt.clusterId
    if (borderPoints.contains(pt.pointId)) newClusterId = borderPoints(pt.pointId)
    mappings.find { _._1.contains(newClusterId) } match {
      case m: Some[(SetOfIds, Long)] => newClusterId = m.get._2
      case _ =>
    }
    if (newClusterId == Int.MinValue) newClusterId = NOISE_LABEL
    pt.withClusterId(newClusterId)
  }

  private def isCorePoint(pt: Point)(implicit settings: Settings): Boolean = pt.neighboursCount >= settings.minPoints
}
