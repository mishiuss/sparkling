package ru.ifmo.rain.distances

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import ru.ifmo.rain.distances.Minkowski.{Chebyshev, Euclidean, Manhattan}
import ru.ifmo.rain.distances.MultiBuilder.{DISTANCES, modalityNorm}
import ru.ifmo.rain.utils.Compares.deq
import ru.ifmo.rain.{ID_COL, Sparkling}

import scala.collection.mutable.ListBuffer
import scala.util.Random

/**
 * Creates multimodal distance metric for given dataset
 * @param df Multimodal dataframe
 */
@Sparkling
class MultiBuilder(df: DataFrame) extends Serializable {
  private val wrappers = ListBuffer[ModalWrapper]()

  /**
   * Appends new modality with meta information
   * @param modality Unique string identifier
   * @param metric String name of intra-modal metric
   * @param dim Dimension of modality's vector representation
   * @param weight Modality's "importance"
   * @return Normalisation coefficient, e.g .the maximum intra-modal distance for dataframe
   */
  @Sparkling
  def newModality(modality: String, metric: String, dim: Int, weight: Double): Double = {
    if (modality.startsWith("_"))
      throw new IllegalArgumentException(s"Modality name should not start with '_', but was '$modality'")
    if (!df.columns.contains(modality))
      throw new IllegalArgumentException(s"Modality '$modality' is not present in dataframe")
    if (wrappers.exists(_.modality == modality))
      throw new IllegalArgumentException(s"Modality '$modality' already considered")
    if (dim <= 0)
      throw new IllegalArgumentException(s"Incorrect dimension value: $dim")
    if (weight <= 0.0) {
      throw new IllegalArgumentException(s"Incorrect weight value: $weight")
    }
    val dist = DISTANCES(metric)
    val norm = modalityNorm(df, modality, dist)
    wrappers += new ModalWrapper(dist, modality, dim, weight, norm)
    norm
  }

  /**
   * Finished  creation of multimodal metric
   * @return MultiDistance with defined previously modalities
   */
  @Sparkling
  def create(): MultiDistance = {
    if (!deq(wrappers.map(_.weight).sum, 1.0))
      throw new IllegalStateException(s"Weights do not sum up to 1.0")
    new MultiDistance(wrappers.sortBy(_.modality).toArray)
  }
}


object MultiBuilder {
  private val DISTANCES = Map[String, ModalDistance](
    "MANHATTAN" -> Manhattan,
    "EUCLIDEAN" -> Euclidean,
    "CHEBYSHEV" -> Chebyshev,
    "COSINE" -> Cosine,
    "CANBERRA" -> Canberra
  )

  /**
   * Calculates normalisation coefficient for the modality
   * @param df Multimodal dataframe
   * @param modality Modality's name
   * @param distance Intra-modal distance for given modality
   * @return Normalisation coefficient, e.g the maximum distance between two objects in dataframe
   */
  def modalityNorm(df: DataFrame, modality: String, distance: ModalDistance): Double = {
    val data = df.rdd.map { row => row.getAs[Long](ID_COL) -> row.getAs[Vector](modality) }
    val init = data.takeSample(withReplacement = false, 2)
    var (p1, p2, p3, p4) = (init(0), init(1), init(0), init(1))

    for (_ <- 0 until 5) {
      p1 = p3
      p2 = p4
      p4 = farthest(data, distance, p1)
      p3 = farthest(data, distance, p4)
      if (p1._1 == p3._1 && p2._1 == p4._1)
        return distance(p1._2, p2._2)
    }
    distance(p3._2, p4._2)
  }

  /**
   * Restores multimodal metric from serialised representation
   * @param serialized Encoded serialised representation
   * @return MultiDistance, defined by serialised string
   */
  @Sparkling
  def fromPy4J(serialized: String): MultiDistance = {
    val wrappers = serialized.split("\n").map { raw =>
      val fields = raw.split(";")
      new ModalWrapper(
        fun = DISTANCES(fields(0)),
        modality = fields(1),
        dim = fields(2).toInt,
        weight = fields(3).toDouble,
        norm = fields(4).toDouble
      )
    }
    new MultiDistance(wrappers.sortBy(_.modality))
  }

  /**
   * Searches for the farthest intra-modal element in the dataframe from the given object
   * @param data Dataframe with single modality remained
   * @param dist Intra-modal distance metric
   * @param x Pivot object
   * @return The farthest from x object in the dataframe
   */
  private def farthest(data: RDD[(Long, Vector)], dist: ModalDistance, x: (Long, Vector)): (Long, Vector) =
    data.map { obj => obj -> dist(obj._2, x._2) }.max()(Ordering.by(_._2))._1

  /**
   * The same method for non-distributed dataframe
   */
  def modalityNorm(localDf: Array[Row], modality: String, distance: ModalDistance): Double = {
    val data = localDf.map { row => row.getAs[Long](ID_COL) -> row.getAs[Vector](modality) }
    val (random, n) = (new Random(), data.length)
    var (p1, p2) = (data(random.nextInt(n)), data(random.nextInt(n)))
    var (p3, p4) = (p1, p2)
    for (_ <- 0 until 5) {
      p1 = p3
      p2 = p4
      p4 = farthest(data, distance, p1)
      p3 = farthest(data, distance, p4)
      if (p1._1 == p3._1 && p2._1 == p4._1)
        return distance(p1._2, p2._2)
    }
    distance(p3._2, p4._2)
  }

  /**
   * The same method for non-distributed dataframe
   */
  private def farthest(localData: Array[(Long, Vector)], dist: ModalDistance, x: (Long, Vector)): (Long, Vector) =
    localData.map { obj => obj -> dist(obj._2, x._2) }.maxBy(_._2)._1
}
