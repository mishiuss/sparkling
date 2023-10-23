package ru.ifmo.rain.measures

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.storage.StorageLevel.{MEMORY_AND_DISK, NONE}
import ru.ifmo.rain.distances.MultiDistance
import ru.ifmo.rain._


/**
 * Implements internal measure functions (cluster validity indices) evaluation,
 * e.g., estimation without knowledge of ground truth labels
 */
object Internals {

  /**
   * Entry point for multimodal dataframe measure evaluation.
   * Removes noise from dataframe and groups objects by their label.
   * Then, passes split clusters into specified evaluator.
   *
   * @param measure String identifier of measure function
   * @param df Multimodal dataframe
   * @param dist Multimodal metric for df
   * @return Computed index value
   */
  @Sparkling
  def evaluate(measure: String, df: DataFrame, dist: MultiDistance): Double = withClearCaches {
    implicit val implicitDist: MultiDistance = dist

    val function = measure match {
      case "ch"    => ch(_, _)
      case "db"    => db(_, _)
      case "db*"   => dbAlter(_, _)
      case "dunn"  => dunn(_, _)
      case "gd31"  => gd31(_, _)
      case "gd41"  => gd41(_, _)
      case "gd41*" => gd41Approx(_, _)
      case "gd51"  => gd51(_, _)
      case "gd51*" => gd51Approx(_, _)
      case "gd33"  => gd33(_, _)
      case "gd43"  => gd43(_, _)
      case "gd53"  => gd53(_, _)
      case "sil"   => sil(_, _)
      case "sil*"  => silApprox(_, _)
      case "sf"    => sf(_, _)
      case _ => throw new IllegalArgumentException(s"Unknown measure: $measure")
    }

    if (!df.columns.contains(LABEL_COL))
      throw new IllegalStateException(s"Dataframe does not contain label column ($LABEL_COL)")
    val (numLabels, cleanDf) = prepareDf(df)
    if (numLabels < 2)
      throw new IllegalStateException(s"Not enough unique labels in dataframe")
    val labelBuckets = Iterator.tabulate(numLabels + 1) { _ - 0.5 }.toArray
    val counts = cleanDf.rdd
      .map { _.getAs[Int](LABEL_COL) }
      .histogram(labelBuckets, evenBuckets = true)
    val labelCounts = (0 until numLabels).zip(counts).toArray
    function.apply(cleanDf, split(cleanDf, labelCounts))
  }

  private def prepareDf(df: DataFrame): (Int, DataFrame) = {
    val labs = df.select(LABEL_COL).distinct().collect.map(_.getInt(0))
    if (labs.contains(NOISE_LABEL)) {
      (labs.length - 1) -> df.filter(df.col(LABEL_COL) =!= NOISE_LABEL)
    } else
      labs.length -> df
  }

  private def split(df: DataFrame, labelCounts: Array[(Int, Long)])(implicit dist: MultiDistance): Array[ClusterInfo] = {
    for ((label, count) <- labelCounts) yield {
      val objects = df.filter { df.col(LABEL_COL) === label }
      if (objects.storageLevel == NONE) objects.persist(MEMORY_AND_DISK)
      new ClusterInfo(objects, count, label)
    }
  }


  /**
   * $$CH(C) = \frac{N-K}{K-1}\cdot\frac{\sum_{c_k\in C}|c_k|\cdot||\overline{c_k}-\overline{X}||}{\sum_{c_k\in C}\sum_{x_i\in c_k}||x_i-\overline{c_k}||}$$
   * The index should increase. Compactness is understood as the distance from points to their centroid,
   * and separability is understood as the distance from the centroid of clusters to the global centroid.
   *
   * @param df Multimodal dataframe
   * @param clusters Grouped into clusters objects
   * @param dist Multimodal metric for df
   * @return Calinski-Harabasz index value
   */
  private def ch(df: DataFrame, clusters: Array[ClusterInfo])(implicit dist: MultiDistance): Double = {
    val (amount, k) = (df.count(), clusters.length)
    val globalCentroid = dist.centroid(df, amount)
    val globalSum = clusters.map { cl => cl.amount * dist(cl.centroid, globalCentroid) }.sum
    val clusterSum = clusters.map { _.distToCentroid.sum() }.sum
    (amount - k).toDouble / (k - 1) * globalSum / clusterSum
  }


  /**
   * $$D(C)=\frac{min_{c_k \in C} \{min_{c_l \in C \setminus c_k} \{\delta(c_k, c_l)\}\}}{max_{c_k \in C} \{\Delta(c_k)\}}$$
   * Generalised Dunn index uses variations of the $$\delta$$ and $$\Delta$$ values:
   * $$ \delta1 (c_k, c_l) = min_{x_i \in c_k, x_j \in c_l} ||x_i-x_j|| $$
   * $$ \delta3 (c_k, c_l) = \frac{1}{|c_k| * |c_l|} \sum_{x_i \in c_l} \sum_{x_j \in c_l} ||x_i - x_j|| $$
   * $$ \delta4 (c_k, c_l) = ||\overline{c_k} - \overline{c_l}|| $$
   * $$ \delta5 (c_k, c_l) = \frac{1}{|c_k| + |c_l|} (\sum_{x_i \in c_k} ||x_i - \overline{c_k}|| + \sum_{x_j \in c_l} ||x_j - \overline{c_l}||) $$
   * $$ \Delta1 (c_k) = max_{x_i,x_j \in c_k} ||x_i - x_j|| $$
   * $$ \Delta3 (c_k) = \frac{2}{|c_k|} \sum_{x_i \in c_k} ||x_i - \overline{c_k}|| $$
   * The index should increase.
   *
   * @param clusters Grouped into clusters objects
   * @param clusterDiam function, that calculates $$\Delta$$, intra-cluster distance
   * @param clusterDist function, that calculates $$\delta$$, inter-cluster distance
   * @return Generalized Dunn index value
   */
  private def generalisedDunn(
                               clusters: Array[ClusterInfo],
                               clusterDiam: ClusterInfo => Double,
                               clusterDist: (ClusterInfo, ClusterInfo) => Double
                             ): Double = {
    var minDistance = Double.MaxValue
    for (x <- 1 until clusters.length; y <- 0 until x) {
      minDistance = math.min(clusterDist(clusters(x), clusters(y)), minDistance)
    }
    minDistance / clusters.map(clusterDiam).max
  }

  private def dunn(df: DataFrame, clusters: Array[ClusterInfo])(implicit dist: MultiDistance): Double =
    generalisedDunn(clusters, dunnDiam1(_), dunnDist1(_, _))

  private def gd31(df: DataFrame, clusters: Array[ClusterInfo])(implicit dist: MultiDistance): Double =
    generalisedDunn(clusters, dunnDiam1(_), dunnDist3(_, _))

  private def gd41(df: DataFrame, clusters: Array[ClusterInfo])(implicit dist: MultiDistance): Double =
    generalisedDunn(clusters, dunnDiam1(_), dunnDist4(_, _))

  /**
   * Uses the $$\Delta1$$ approximation, where $$\Delta1$$ is calculated as the distance from the point farthest from
   * the centroid to the point farthest from it. This allows for a significant reduction in computational complexity.
   */
  private def gd41Approx(df: DataFrame, clusters: Array[ClusterInfo])(implicit dist: MultiDistance): Double =
    generalisedDunn(clusters, dunnDiam1Approx(_), dunnDist4(_, _))

  private def gd51(df: DataFrame, clusters: Array[ClusterInfo])(implicit dist: MultiDistance): Double =
    generalisedDunn(clusters, dunnDiam1(_), dunnDist5(_, _))

  /**
   * Uses the $$\Delta1$$ approximation, where $$\Delta1$$ is calculated as the distance from the point farthest from
   * the centroid to the point farthest from it. This allows for a significant reduction in computational complexity.
   */
  private def gd51Approx(df: DataFrame, clusters: Array[ClusterInfo])(implicit dist: MultiDistance): Double =
    generalisedDunn(clusters, dunnDiam1Approx(_), dunnDist5(_, _))

  private def gd33(df: DataFrame, clusters: Array[ClusterInfo])(implicit dist: MultiDistance): Double =
    generalisedDunn(clusters, dunnDiam3(_), dunnDist3(_, _))

  private def gd43(df: DataFrame, clusters: Array[ClusterInfo])(implicit dist: MultiDistance): Double =
    generalisedDunn(clusters, dunnDiam3(_), dunnDist4(_, _))

  private def gd53(df: DataFrame, clusters: Array[ClusterInfo])(implicit dist: MultiDistance): Double =
    generalisedDunn(clusters, dunnDiam3(_), dunnDist5(_, _))

  private def dunnDiam1(cluster: ClusterInfo)(implicit dist: MultiDistance): Double = {
    val data = cluster.data.rdd
    data.cartesian(data).map { pair => dist(pair._1, pair._2) }.max()
  }

  private def dunnDiam1Approx(clusterInfo: ClusterInfo)(implicit dist: MultiDistance): Double = {
    val (c, data) = (clusterInfo.centroid, clusterInfo.data.rdd)
    val p = dist.farthest(data, c)
    dist(p, dist.farthest(data, p))
  }

  private def dunnDiam3(cluster: ClusterInfo)(implicit dist: MultiDistance): Double = {
    2.0 * cluster.distToCentroid.sum() / cluster.amount
  }

  private def dunnDist1(x: ClusterInfo, y: ClusterInfo)(implicit dist: MultiDistance): Double = {
    x.data.rdd.cartesian(y.data.rdd).map { pair => dist(pair._1, pair._2) }.min()
  }

  private def dunnDist3(x: ClusterInfo, y: ClusterInfo)(implicit dist: MultiDistance): Double = {
    val mul = 0.5 / (x.amount * y.amount)
    mul * x.data.rdd.cartesian(y.data.rdd).map { pair => dist(pair._1, pair._2) }.sum()
  }

  private def dunnDist4(x: ClusterInfo, y: ClusterInfo)(implicit dist: MultiDistance): Double = {
    dist(x.centroid, y.centroid)
  }

  private def dunnDist5(x: ClusterInfo, y: ClusterInfo)(implicit dist: MultiDistance): Double = {
    (x.distToCentroid.sum() + y.distToCentroid.sum()) / (x.amount + y.amount)
  }


  /**
   * $$ Sil(C) = \frac{1}{N} \sum_{c_k \in C} \sum_{x_i \in c_k} \frac{b(x_i, c_k) - a(x_i, c_k)}{max\{a(x_i, c_k), b(x_i, c_k)\}} $$
   * $$ a(x_i, c_k) = \frac{1}{|c_k|} \sum_{x_j \in c_k} ||x_i - x_j|| $$ is average intra-cluster distance
   * $$ b(x_i, c_k) = min_{c_l \in C \setminus c_k} \{\frac{1}{|c_l|} \sum_{x_j \in c_l} ||x_i -x_j||\} $$ is average inter-cluster distance
   * $$ -1 \leq Sil(C) \leq 1 $$
   * The index should increase.
   *
   * @param df Multimodal dataframe
   * @param clusters Grouped into clusters objects
   * @param dist Multimodal metric for df
   * @return Silhouette value
   */
  private def sil(df: DataFrame, clusters: Array[ClusterInfo])(implicit dist: MultiDistance): Double = {
    val labelToAmount = clusters.map { cl => cl.label -> cl.amount }.toMap
    val broadcastAmounts = df.rdd.context.broadcast(labelToAmount)
    val baValues = clusters.map { cl => (silB(df, cl, broadcastAmounts), silA(cl)) }
    baValues.map(silCombine).sum / df.count()
  }

  private def silA(cluster: ClusterInfo)(implicit dist: MultiDistance): RDD[(Long, Double)] = {
    val (size, data) = (cluster.amount, cluster.data.rdd)
    data.cartesian(data)
      .map { case (x, y) => x.getAs[Long](ID_COL) -> dist(x, y) }
      .reduceByKey(_ + _)
      .mapValues { _ / size }
  }

  private def silB(df: DataFrame, cluster: ClusterInfo, amounts: Broadcast[Map[Int, Long]])
                  (implicit dist: MultiDistance): RDD[(Long, Double)] = {
    val another = df.filter { df.col(LABEL_COL) =!= cluster.label }
    cluster.data.rdd.cartesian(another.rdd).map { case (x, y) =>
      (x.getAs[Long](ID_COL), y.getAs[Int](LABEL_COL)) -> dist(x, y)
    }.reduceByKey(_ + _).map {
      case ((id, label), sumDist) => (id, sumDist / amounts.value(label))
    }.reduceByKey(math.min)
  }

  private def silCombine(ba: (RDD[(Long, Double)], RDD[(Long, Double)])): Double = {
    ba._1.join(ba._2).map { case (_, (b, a)) => (b - a) / math.max(b, a) }.sum()
  }


  /**
   * Approximation of Silhouette index. To reduce computational complexity,
   * *a* and *b* are calculated through centroids, and not through all pairwise distances.
   *
   * @param df Multimodal dataframe
   * @param clusters Grouped into clusters objects
   * @param dist Multimodal metric for df
   * @return Approximated Silhouette value
   */
  private def silApprox(df: DataFrame, clusters: Array[ClusterInfo])(implicit dist: MultiDistance): Double = {
    val centroids = clusters.map(c => c.label -> c.centroid).toMap
    val broadcastCentroids = df.sparkSession.sparkContext.broadcast(centroids)
    df.rdd.map { obj =>
      val label = obj.getAs[Int](LABEL_COL)
      var (a, b) = (0.0, Double.MaxValue)
      for (cent <- broadcastCentroids.value) {
        val d = dist(obj, cent._2)
        if (cent._1 == label) a = d
        else b = math.min(b, d)
      }
      (b - a) / math.max(b, a)
    }.sum() / df.count()
  }


  /**
   * Compactness is understood as the distance from cluster objects to their centroids,
   * and separability is understood as the distance between centroids.
   * $$ DB(C) = \frac{1}{K} \sum_{c_k \in C} max_{c_l \in C \setminus c_k} \{\frac{S(c_k) + S(c_l)} {||\overline{c_k} - \overline{c_l}||}\} $$
   * $$ S(c_k) = \frac{1}{|c_k|} \sum_{x_i \in c_k} ||x_i - \overline{c_k}|| $$
   * The index should decrease.
   *
   * @param df Multimodal dataframe
   * @param clusters Grouped into clusters objects
   * @param dist Multimodal metric for df
   * @return Davies-Bouldin index value
   */
  private def db(df: DataFrame, clusters: Array[ClusterInfo])(implicit dist: MultiDistance): Double = {
    val means = clusters.map(_.distToCentroid.mean())
    var clustersSum = 0.0
    for (x <- clusters.indices) {
      val sep = for (y <- clusters.indices; if x != y)
        yield (means(x) + means(y)) / dist(clusters(x).centroid, clusters(y).centroid)
      clustersSum += sep.max
    }
    clustersSum / clusters.length
  }

  /**
   * Compactness is understood as the distance from cluster objects to their centroids,
   * and separability is understood as the distance between centroids.
   * $$ DB*(C) = \frac{1}{K} \sum_{c_k \in C} \frac{max_{c_l \in C \setminus c_k} \{S(c_k) + S(c_l)\}} {min_{c_l \in C \setminus c_k}\{||\overline{c_k} - \overline{c_l}||\}} $$
   * The index should decrease.
   *
   * @param df Multimodal dataframe
   * @param clusters Grouped into clusters objects
   * @param dist Multimodal metric for df
   * @return Davies-Bouldin* index value
   */
  private def dbAlter(df: DataFrame, clusters: Array[ClusterInfo])(implicit dist: MultiDistance): Double = {
    val means = clusters.map(_.distToCentroid.mean())
    var clustersSum = 0.0
    for (x <- clusters.indices) {
      var (sep, comp) = (Double.MinValue, Double.MaxValue)
      for (y <- clusters.indices; if x != y) {
        sep = math.max(means(x) + means(y), sep)
        val cDist = dist(clusters(x).centroid, clusters(y).centroid)
        comp = math.min(cDist, comp)
      }
      clustersSum += sep / comp
    }
    clustersSum / clusters.length
  }

  private def compact(cluster: ClusterInfo)(implicit p: Double): Double = {
    val quantile = ((1.0 - p) * cluster.amount).toLong
    val indexed = cluster.distToCentroid.sortBy(identity).zipWithIndex().map { _.swap }
    indexed.filterByRange(quantile, cluster.amount).values.sum()
  }

  /**
   * $$ SF(C) = \frac{\sum_{c_k \in C} min_{c_l \in C \setminus c_k} \{||\overline{c_k} - \overline{c_l}||\}} {\sum_{c_k \in C} 10/|c_k| \sum max_{x_i \in c_k}(0.1 * |c_k|) * ||\overline{x_i} - \overline{c_k}||} $$
   * The index should increase.
   *
   * @param df Multimodal dataframe
   * @param clusters Grouped into clusters objects
   * @param dist Multimodal metric for df
   * @return SF index value
   */
  private def sf(df: DataFrame, clusters: Array[ClusterInfo])(implicit dist: MultiDistance): Double = {
    implicit val p: Double = 0.1
    var separation = 0.0
    for (idx <- clusters.indices) {
      val sep = for (other <- clusters.indices if other != idx)
        yield dist(clusters(idx).centroid, clusters(other).centroid)
      separation += sep.min
    }
    separation / clusters.map(compact).sum
  }
}