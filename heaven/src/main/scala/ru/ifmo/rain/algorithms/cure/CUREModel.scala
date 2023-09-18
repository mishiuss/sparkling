package ru.ifmo.rain.algorithms.cure

import org.apache.spark.sql.{DataFrame, Row}
import ru.ifmo.rain.{LABEL_FIELD, Sparkling}
import ru.ifmo.rain.algorithms.ClusteringModel
import ru.ifmo.rain.algorithms.ClusteringModel.withLabel


@Sparkling
class CUREModel(private val kdTree: KDTree) extends ClusteringModel {

  @Sparkling
  override def predict(obj: Row): Int =
    kdTree.closestPointOfOtherCluster(KDPoint(obj)).cluster.id

  @Sparkling
  override def predict(df: DataFrame): DataFrame = {
    val (ss, schema) = (df.sparkSession, df.schema)
    val broadcastTree = ss.sparkContext.broadcast(kdTree)
    val result = df.rdd.map { obj =>
      val (tree, p) = (broadcastTree.value, KDPoint(obj))
      val representative = tree.closestPointOfOtherCluster(p)
      withLabel(obj, representative.cluster.id)
    }
    ss.createDataFrame(result, schema.add(LABEL_FIELD))
  }
}
