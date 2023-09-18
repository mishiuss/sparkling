package ru.ifmo.rain.algorithms.kmeans

import org.apache.spark.sql.{DataFrame, Row}
import ru.ifmo.rain.{LABEL_COL, LABEL_FIELD, Sparkling}
import ru.ifmo.rain.algorithms.ClusteringModel
import ru.ifmo.rain.algorithms.ClusteringModel.withLabel
import ru.ifmo.rain.distances.MultiDistance


@Sparkling
class KMeansModel(
                   private val kCentroids: Array[Row],
                   private val distance: MultiDistance
                 ) extends ClusteringModel {

  private val labeledCentroids = kCentroids.zipWithIndex

  @Sparkling
  def centroids(): Array[Row] = kCentroids

  @Sparkling
  override def predict(obj: Row): Int = labeledCentroids.minBy { ci => distance(ci._1, obj) }._2

  @Sparkling
  override def predict(df: DataFrame): DataFrame = {
    val (ss, schema) = (df.sparkSession, df.schema)
    if (schema.fields.exists(_.name == LABEL_COL)) {
      throw new IllegalArgumentException("Input dataframe has been already labeled")
    }
    val (centroids, dist) = (ss.sparkContext.broadcast(labeledCentroids), distance)
    val result = df.rdd.map { obj =>
      withLabel(obj, centroids.value.minBy { ci => dist(ci._1, obj) }._2)
    }
    ss.createDataFrame(result, schema.add(LABEL_FIELD))
  }
}
