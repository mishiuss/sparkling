package ru.ifmo.rain.algorithms.spectral

import org.apache.spark.sql.{DataFrame, Row}
import ru.ifmo.rain.Sparkling
import ru.ifmo.rain.algorithms.ClusteringModel


@Sparkling
class SpectralModel(private val clustered: DataFrame) extends ClusteringModel {

  @Sparkling
  def dataframe(): DataFrame = clustered

  override def predict(obj: Row): Int = throw new NotImplementedError()

  override def predict(df: DataFrame): DataFrame = throw new NotImplementedError()
}
