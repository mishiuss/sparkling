package ru.ifmo.rain.algorithms.meanshift

import org.apache.spark.sql.{DataFrame, Row}
import ru.ifmo.rain.{LABEL_COL, LABEL_FIELD, NOISE_LABEL, Sparkling}
import ru.ifmo.rain.algorithms.ClusteringModel
import ru.ifmo.rain.algorithms.ClusteringModel.withLabel
import ru.ifmo.rain.distances.MultiDistance


@Sparkling
class MeanShiftModel(
                      private val shiftedMeans: Array[Row],
                      private val distance: MultiDistance,
                      private val bandwidth: Double
                    ) extends ClusteringModel {

  private val labeledMeans = shiftedMeans.zipWithIndex
  private var noise = false

  @Sparkling
  def setNoise(noise: Boolean): MeanShiftModel = { this.noise = noise; this }

  @Sparkling
  def means(): Array[Row] = shiftedMeans

  @Sparkling
  override def predict(obj: Row): Int = {
    val nearest = labeledMeans.map { case (pivot, label) => label -> distance(obj, pivot) }.minBy(_._2)
    if (noise && nearest._2 >= bandwidth) NOISE_LABEL else nearest._1
  }

  @Sparkling
  override def predict(df: DataFrame): DataFrame = {
    val (ss, schema) = (df.sparkSession, df.schema)
    if (schema.fields.exists(_.name == LABEL_COL)) {
      throw new IllegalArgumentException("Input dataframe has been already labeled")
    }
    val pivots = ss.sparkContext.broadcast(labeledMeans)
    val (withNoise, dist, radius) = (noise, distance, bandwidth)
    val result = df.rdd.map { obj =>
      val nearest = pivots.value.map { case (pivot, label) => label -> dist(obj, pivot) }.minBy(_._2)
      withLabel(obj, if (withNoise && nearest._2 >= radius) NOISE_LABEL else nearest._1)
    }
    ss.createDataFrame(result, schema.add(LABEL_FIELD))
  }
}