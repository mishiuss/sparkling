package ru.ifmo.rain.algorithms.bisecting

import org.apache.spark.sql.{DataFrame, Row}
import ru.ifmo.rain.{LABEL_FIELD, Sparkling}
import ru.ifmo.rain.algorithms.ClusteringModel
import ru.ifmo.rain.algorithms.ClusteringModel.withLabel
import ru.ifmo.rain.distances.MultiDistance


@Sparkling
class BisectingKMeansModel(
                            private val root: ClusteringTreeNode,
                            private val dist: MultiDistance
                          ) extends ClusteringModel {

  private val leafCentroids = root.leafNodes.map(_.center)

  @Sparkling
  def centroids(): Array[Row] = leafCentroids

  @Sparkling
  override def predict(obj: Row): Int = {
    root.predict(obj, 0.0)(dist)._1
  }

  @Sparkling
  override def predict(df: DataFrame): DataFrame = {
    val (ss, schema) = (df.sparkSession, df.schema)
    val broadcastRoot = ss.sparkContext.broadcast(root)
    val result = df.rdd.map { obj =>
      withLabel(obj, broadcastRoot.value.predict(obj, 0.0)(dist)._1)
    }
    ss.createDataFrame(result, schema.add(LABEL_FIELD))
  }
}
