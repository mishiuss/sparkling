package ru.ifmo.rain.algorithms.clique

import org.apache.spark.sql.{DataFrame, Row}
import ru.ifmo.rain.algorithms.ClusteringModel
import ru.ifmo.rain.algorithms.ClusteringModel.withLabel
import ru.ifmo.rain.algorithms.clique.CLIQUEModel.{DenseUnits, labelForObj}
import ru.ifmo.rain.distances.MultiDistance
import ru.ifmo.rain.{LABEL_FIELD, NOISE_LABEL, Sparkling}


@Sparkling
class CLIQUEModel(
                   private val denseUnits: IndexedSeq[DenseUnits],
                   private val splits: Int,
                   private val dist: MultiDistance
                 ) extends ClusteringModel {
  @Sparkling
  override def predict(obj: Row): Int = labelForObj(obj, denseUnits, splits, dist)

  @Sparkling
  override def predict(df: DataFrame): DataFrame = {
    val (s, ss) = (splits, df.sparkSession)
    val broadcastUnits = ss.sparkContext.broadcast(denseUnits)
    val labeled = df.rdd.map { obj => withLabel(obj, labelForObj(obj, broadcastUnits.value, s, dist)) }
    ss.createDataFrame(labeled, df.schema.add(LABEL_FIELD))
  }

}

object CLIQUEModel {
  private[clique] type DenseUnit = Map[Int, Int]
  private[clique] type DenseUnits = Array[DenseUnit]

  private def labelForObj(obj: Row, units: IndexedSeq[DenseUnits], splits: Int, dist: MultiDistance): Int = {
    for (clusterId <- units.indices) {
      for (unit <- units(clusterId)) {
        val objInsideUnit = unit.forall { case (globalIdx, bucket) =>
          val value = dist.byGlobal(obj, globalIdx)
          (value * splits % splits).floor.toInt == bucket
        }
        if (objInsideUnit) return clusterId
      }
    }
    NOISE_LABEL
  }
}