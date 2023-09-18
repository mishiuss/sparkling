package ru.ifmo.rain.algorithms.sandbox

import org.apache.spark.ml.feature.{MinMaxScaler, VectorAssembler}
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK
import ru.ifmo.rain.algorithms.sandbox.Sandbox.ss
import ru.ifmo.rain.distances.MultiBuilder
import ru.ifmo.rain.{CLASS_COL, ID_COL, logger}

abstract class UniModalSandbox extends Sandbox {
  val datasets: Seq[String] = List(
    "abalone",
    "blocks",
    "character",
    "iris",
    "mfeature",
    "visualizing_soil",
    "volcanoes-d4",
    "wall-robot-navigation",
    "wine-quality-white"
  )

  private def getSetup(name: String): TestSetup = {
    val raw = ss.read.option("header", "true").option("inferSchema", "true").csv(s"src/test/data/$name.csv")
    val features = raw.columns.filter(_ != "class")
    val vector = new VectorAssembler().setInputCols(features).setOutputCol("vector").transform(raw)
    val minMax = new MinMaxScaler().setMin(-1.0).setMax(1.0)
      .setInputCol("vector").setOutputCol("features")
    val df = minMax.fit(vector).transform(vector)
      .select("features", "class")
      .withColumn(ID_COL, monotonically_increasing_id())
      .withColumnRenamed("class", CLASS_COL)
      .repartition(ss.sparkContext.defaultParallelism)
      .persist(MEMORY_AND_DISK)

    val builder = new MultiBuilder(df)
    builder.newModality("features", "EUCLIDEAN", features.length, 1.0)
    new TestSetup(df, df.count(), builder.create())
  }

  private var current: (String, TestSetup) = "fake" -> null

  private def supplier(name: String): TestSetup = {
    if (current._1 != name) {
      logger.info(s"Loading setup for $name")
      current = name -> getSetup(name)
    }
    current._2
  }

  datasets.foreach { name => forSetup(name)(supplier) }
}