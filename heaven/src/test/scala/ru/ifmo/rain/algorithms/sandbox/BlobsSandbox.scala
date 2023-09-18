package ru.ifmo.rain.algorithms.sandbox

import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types.{IntegerType, LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK
import ru.ifmo.rain.algorithms.sandbox.Sandbox.ss
import ru.ifmo.rain.distances.Minkowski.{Euclidean, Manhattan}
import ru.ifmo.rain.distances.{ModalWrapper, MultiBuilder, MultiDistance}
import ru.ifmo.rain.{CLASS_COL, ID_COL, logger}

import scala.util.Random
import scala.util.matching.Regex


class Blobs(private val proxyWrappers: Array[ModalWrapper]) {
  private val fields = proxyWrappers.map { wrapper =>
    StructField(wrapper.modality, VectorType, nullable = false)
  }.toList

  private val schema = new StructType((
    StructField(ID_COL, LongType, nullable = false)
      :: StructField(CLASS_COL, IntegerType, nullable = false)
      :: fields
    ).toArray)

  private val random = new Random(42)

  def generate(n: Long, centers: Int, ss: SparkSession): (DataFrame, MultiDistance) = {
    val centroids = for (_ <- 0 until centers) yield centroid()
    val variances = for (_ <- 0 until centers) yield math.random * 0.05 + 0.001
    val sizesRatio = for (_ <- 0 until centers) yield math.random + 0.1
    val sizes = sizesRatio.map(_ / sizesRatio.sum).toArray

    val local = for (id <- 0L until n) yield {
      val pivotIdx = takePivot(sizes)
      from(centroids(pivotIdx), variances(pivotIdx), id, pivotIdx)
    }
    val partitions = math.sqrt(n) / math.log(n)
    val rdd = ss.sparkContext.parallelize(local, partitions.toInt)
    val df = ss.createDataFrame(rdd, schema)
    val normalised = fields.foldLeft(df) { case (dataframe, field) =>
      val tmpName = s"_${field.name}"
      val minMax = new MinMaxScaler().setMin(-1.0).setMax(1.0)
        .setInputCol(field.name).setOutputCol(tmpName)
      val norm = minMax.fit(dataframe).transform(dataframe)
      norm.drop(field.name).withColumnRenamed(tmpName, field.name)
    }.persist(MEMORY_AND_DISK)
    val builder = new MultiBuilder(normalised)
    proxyWrappers.foreach { wrapper =>
      val metric = wrapper.fun.getClass.getSimpleName.toUpperCase.replace("$", "")
      builder.newModality(wrapper.modality, metric, wrapper.dim, wrapper.weight)
    }
    (normalised, builder.create())
  }

  private def centroid(): Array[(ModalWrapper, Array[Double])] =
    proxyWrappers.map { wrapper =>
      val values = for (_ <- 0 until wrapper.dim) yield math.random
      wrapper -> values.toArray
    }

  private def from(pivot: Array[(ModalWrapper, Array[Double])], variance: Double, id: Long, label: Int): Row = {
    val modalities = pivot.map { wrapper =>
      val biased = wrapper._2.map { v => v + (random.nextDouble() - 0.5) * variance }
      Vectors.dense(biased)
    }
    new GenericRowWithSchema((Seq[Any](id, label) ++ modalities).toArray, schema)
  }

  private def takePivot(sizes: Array[Double]): Int = {
    val r = math.random
    var (idx, curWeight) = (0, 0.0)
    while (idx < sizes.length && curWeight < r) {
      curWeight += sizes(idx)
      idx += 1
    }
    idx - 1
  }
}


abstract class BlobsSandbox extends Sandbox {
  private val m1 = new ModalWrapper(Manhattan,  "m1", 30, 0.4, 0.0)
  private val m2 = new ModalWrapper(Euclidean, "m2", 20, 0.6, 0.0)
  protected val generator = new Blobs(Array(m1, m2))

  protected val nValues: Seq[Long] = List(100000L, 500000L, 2500000L)
  protected val centerValues: Seq[Int] = List(2, 5, 11, 23)

  private var current: (String, TestSetup) = "fake" -> null

  private def supplier(name: String): TestSetup = {
    if (current._1 != name) {
      logger.info(s"Loading setup for $name")
      val strN = new Regex("n=\\d+").findFirstMatchIn(name).get
      val strC = new Regex("c=\\d+").findFirstMatchIn(name).get
      val n = strN.matched.substring(2).toLong
      val c = strC.matched.substring(2).toInt
      val (df, dist) = generator.generate(n, c, ss)
      current = name -> new TestSetup(df, n, dist)
    }
    current._2
  }

  nValues.foreach { n =>
    centerValues.foreach { c =>
      forSetup(s"blobs: n=$n, c=$c")(supplier)
    }
  }
}
