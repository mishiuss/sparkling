package ru.ifmo.rain.utils

import org.apache.spark.sql.DataFrame
import ru.ifmo.rain.distances.MultiDistance
import ru.ifmo.rain.{ID_COL, Sparkling}

@Sparkling
object Meta {

  /**
   * Extract meta-features of given dataframe in order to suggest recommendation by CVI Predictor.
   * Calculates all pairwise distances between samples. Normalises them to upper bound 1.0.
   * Extract statistical values: [mean, variance, standard deviation, skewness, kurtosis].
   * Next 10 values represents percentage of distances in ranges [0.0, 0.1), [0.1, 0.2), ... [0.9, 1.0];
   * To calculate the last 4 meta-features, distances values forced to obtain zero mean and unit variance;
   * after that count percentage of values in ranges [0, 1), [1, 2), [2, 3) and other positives
   *
   * @param df Multimodal dataframe
   * @param dist Multimodal metric for df
   * @param amount Number of objects in dataframe
   * @return Array of meta-features
   */
  @Sparkling
  def evaluate(df: DataFrame, dist: MultiDistance, amount: Long): Array[Double] = {
    val distances = df.rdd.cartesian(df.rdd)
      .filter { case (x, y) => x.getAs[Long](ID_COL) < y.getAs[Long](ID_COL) }
      .map { case (x, y) => dist(x, y) }

    val n = amount * (amount - 1) / 2L
    val mean = distances.sum() / n

    val (m2sum, m3sum, m4sum) = distances.map { d =>
      val diff = d - mean
      val m2 = diff * diff
      val m3 = m2 * diff
      val m4 = m3 * diff
      (m2, m3, m4)
    }.reduce { case (x, y)  =>
      (x._1 + y._1, x._2 + y._2, x._3 + y._3)
    }
    val variance = m2sum / n
    val std = math.sqrt(variance)

    val skew = m3sum / n / (std * std * std)
    val kurt = m4sum / n / (std * std * std * std) - 3

    val mdBuckets = Iterator.tabulate(11) { _ * 0.1 }.toArray
    val mdPercentage = distances.histogram(mdBuckets, evenBuckets = true).map(_ / n.toDouble)

    val centered = distances.map { d => (d - mean) / std }
    val cBuckets = Array(0.0, 1.0, 2.0, 3.0, Double.MaxValue)
    val cPercentage = centered.histogram(cBuckets).map(_ / n.toDouble)

    (List(mean, variance, std, skew, kurt) ::: mdPercentage.toList ::: cPercentage.toList).toArray
  }
}
