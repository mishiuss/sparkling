package ru.ifmo.rain.distances

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.VectorUDT
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row}
import ru.ifmo.rain.utils.BLAS

import scala.math.{abs, sqrt}


/**
 * Algebraic operations on multimodal objects and distance metric between them
 * @param modals array of modality's meta information
 */
class MultiDistance(private val modals: Array[ModalWrapper]) extends Serializable {
  private val schema = new StructType(
    modals.map { wrapper => StructField(wrapper.modality, new VectorUDT(), nullable = false) }
  )

  private def extract(modality: String, obj: Row): Vector = obj.getAs[Vector](modality)

  private def extract(modality: String, xObj: Row, yObj: Row): (Vector, Vector) =
    (extract(modality, xObj), extract(modality, yObj))

  def modalities: Array[ModalWrapper] = modals

  /**
   * Extracts sequence of vector representations from given multimodal object
   * @param obj Multimodal object
   * @return List of underlying modalities vectors with meta information
   */
  def modalities(obj: Row): Array[(ModalWrapper, Vector)] = {
    modals.map { wrapper =>
      val vector = extract(wrapper.modality, obj)
      assert(vector.size == wrapper.dim)
      wrapper -> vector
    }
  }

  /**
   * Extracts sequence of vector representations from given multimodal objects
   * @param xObj First multimodal object
   * @param yObj Second multimodal object
   * @return List of underlying modalities vectors with meta information
   */
  def modalities(xObj: Row, yObj: Row): Array[(ModalWrapper, (Vector, Vector))] =
    modals.map { wrapper =>
      val (x, y) = extract(wrapper.modality, xObj, yObj)
      assert(x.size == y.size && x.size == wrapper.dim)
      wrapper -> (x, y)
    }

  /**
   * Total amount of dimensions
   */
  val globalDim: Int = modals.map(_.dim).sum

  private def byGlobalWithIndex(obj: Row, index: Int): (ModalWrapper, Int, Double) = {
    var bound = 0
    for (wrapper <- modals) {
      if (index < bound + wrapper.dim) {
        val local = index - bound
        val value = extract(wrapper.modality, obj)(local)
        return (wrapper, local, value)
      }
      bound += wrapper.dim
    }
    throw new IllegalArgumentException("Global index is out of bounds")
  }

  def byGlobal(obj: Row, index: Int): Double = byGlobalWithIndex(obj, index)._3

  /**
   * Converts multimodal object's values into single array
   * @param obj Multimodal object
   * @return Array of "spliced" object's values
   */
  def flatten(obj: Row): Array[Double] = {
    val flatValues = Array.fill(globalDim) { 0.0 }
    var bound = 0
    for (wrapper <- modals) {
      extract(wrapper.modality, obj).foreachActive {
        case (idx, value) => flatValues(bound + idx) = value
      }
      bound += wrapper.dim
    }
    flatValues
  }

  /**
   * Splits array of values into modalities' vectors
   * @param flatValues "Spliced" multimodal object's values
   * @return Multimodal object with corresponding modalities structure
   */
  def wrap(flatValues: Array[Double]): Row = {
    assert(flatValues.length == globalDim)
    var bound = 0
    val vectors = for (wrapper <- modals) yield {
      val start = bound
      bound += wrapper.dim
      Vectors.dense(flatValues.slice(start, bound))
    }
    new GenericRowWithSchema(vectors.toArray, schema)
  }

  /**
   * Distance metric between multimodal objects.
   * $$ D(X, Y) = \sqrt{d_1(x_1, y_1)^2 + d_2(x_2, y_2)^2 + \dots + d_n(x_n, y_n)^2} $$
   * $$ d_1, d_2, \dots, d_n $$ - intramodal distance functions
   * $$ x_1, x_2, \dots, x_n $$ - modalities vectors of X multimodal object
   * $$ y_1, y_2, \dots, y_n $$ - modalities vectors of Y multimodal object
   * @param xObj First multimodal object
   * @param yObj Second multimodal object
   * @return Metric of objects dissimilarity
   */
  def apply(xObj: Row, yObj: Row): Double = {
    val modalDistances = modalities(xObj, yObj).map { case (wrapper, (x, y)) =>
      val d = wrapper(x, y)
      wrapper.weight * d * d
    }
    sqrt(modalDistances.sum)
  }

  def approxDistByGlobal(xObj: Row, yObj: Row, index: Int): Double = {
    val (wrapper, local, x) = byGlobalWithIndex(xObj, index)
    val y = extract(wrapper.modality, yObj)(local)
    sqrt(wrapper.weight) * abs(x - y) / wrapper.norm
  }

  def approxDistByGlobal(obj: Row, dimValue: Double, index: Int): Double = {
    val (wrapper, _, x) = byGlobalWithIndex(obj, index)
    sqrt(wrapper.weight) * abs(x - dimValue) / wrapper.norm
  }

  def apply(modality: String): ModalWrapper = modals.find(_.modality == modality).get

  def zeroObj: Row = {
    val zeros = modalities.map { wrapper => Vectors.sparse(wrapper.dim, Seq()) }
    new GenericRowWithSchema(zeros.toArray, schema)
  }

  def generate(generator: ModalWrapper => Vector): Row = {
    val values = modalities.map(generator)
    new GenericRowWithSchema(values.toArray, schema)
  }

  def l2(obj: Row): Double =
    modalities(obj).map { case (wrapper, vector) => BLAS.l2(vector) * wrapper.weight }.sum

  def centroid(data: RDD[Row], amount: Long): Row = {
    if (amount == 0L || data.isEmpty()) throw new IllegalArgumentException("Trying to calc centroid on empty data")
    scale(data.reduce {
      plus(_, _)
    }, 1.0 / amount)
  }

  def centroid(df: DataFrame, amount: Long): Row = centroid(df.rdd, amount)

  def centroid(data: Iterable[Row]): Row = {
    if (data.isEmpty) throw new IllegalArgumentException("Trying to calc centroid on empty data")
    scale(data.reduce {
      plus(_, _)
    }, 1.0 / data.size)
  }

  /**
   * Basic scalar multiplication operation
   * $$ aX = [ax_1, ax_2, \dots, ax_n] $$
   * $$ x_1, x_2, \dots, x_n $$ - modalities vectors of X multimodal object
   * $$ [z_1, z_2, \dots, z_3] $$ - concatenation of vectors
   * @param obj Mutlimodal object
   * @param a Coefficient of multiplication
   * @return New multiplied multimodal object
   */
  def scale(obj: Row, a: Double): Row = {
    val vectors = modalities(obj).map { mv => BLAS.scale(mv._2, a) }
    new GenericRowWithSchema(vectors.toArray, schema)
  }

  def farthest(data: RDD[Row], pivot: Row): Row =
    data.map { obj => (obj, apply(obj, pivot)) }.max()(Ordering.by(_._2))._1

  def agg(xObj: Row, yObj: Row, f: (Double, Double) => Double): Row = {
    val vectors = modalities(xObj, yObj).map({ case (_, (x, y)) => BLAS.aggregate(x, y, f) })
    new GenericRowWithSchema(vectors.toArray, schema)
  }

  /**
   * Basic summation operation for two multimodal objects
   * $$ aX + bY = [ax_1 + by_1, ax_2 + by_2, \dots, ax_3 + by_3] $$
   * $$ x_1, x_2, \dots, x_n $$ - modalities vectors of X multimodal object
   * $$ y_1, y_2, \dots, y_n $$ - modalities vectors of Y multimodal object
   * $$ [z_1, z_2, \dots, z_3] $$ - concatenation of vectors
   * @param xObj First multimodal object
   * @param yObj Second multimodal object
   * @param a Coefficient for first object, 1.0 by default
   * @param b Coefficient for second object, 1.0 by default
   * @return New multimodal object, considered as multimodal summation result
   */
  def plus(xObj: Row, yObj: Row, a: Double = 1.0, b: Double = 1.0): Row = {
    val vectors = modalities(xObj, yObj).map({ case (_, (x, y)) => BLAS.plus(a, x, b, y) })
    new GenericRowWithSchema(vectors.toArray, schema)
  }
}