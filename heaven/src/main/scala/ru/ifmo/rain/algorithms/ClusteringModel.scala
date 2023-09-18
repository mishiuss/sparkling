package ru.ifmo.rain.algorithms

import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.{DataFrame, Row}
import ru.ifmo.rain.LABEL_FIELD


/**
 * Base class for fitted clustering models, produced by clustering algorithms
 */
abstract class ClusteringModel extends Serializable {

  /**
   * Compute label for a single multimodal object
   * @param obj Multimodal object of the same structure, as dataframe used to fit model
   * @return label value in range [0, num clusters)
   */
  def predict(obj: Row): Int

  /**
   * Compute label for each object in a multimodal dataframe
   * @param df Multimodal dataframe of the same structure, as dataframe used to fit model
   * @return dataframe with label column, values in range [0, num clusters)
   */
  def predict(df: DataFrame): DataFrame
}


object ClusteringModel {
  def withLabel(obj: Row, label: Int): Row = {
    val newSchema = obj.schema.add(LABEL_FIELD)
    new GenericRowWithSchema((obj.toSeq :+ label).toArray, newSchema)
  }
}
