package ru.ifmo.rain.distances

import org.apache.spark.mllib.linalg.VectorUDT
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.ml.linalg.Vectors.{dense, sparse}
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types.{StructField, StructType}
import ru.ifmo.rain.utils.Compares.deq

import scala.math.{min, sqrt}


class MultiDistanceTest extends ModalDistanceTest {
  private val schema = new StructType(Array(
    StructField("dd", new VectorUDT(), nullable = false),
    StructField("ss", new VectorUDT(), nullable = false),
    StructField("ds", new VectorUDT(), nullable = false),
    StructField("sd", new VectorUDT(), nullable = false)
  ))

  private val xRow = new GenericRowWithSchema(Array(dx, sx, dw, sz), schema)
  private val yRow = new GenericRowWithSchema(Array(dy, sy, sw, dz), schema)

  private val components = Array(
    mockModality("dd", 1.4, 5, 0.5, 2.0),
    mockModality("ss", 1.8, 7, 0.1, 3.0),
    mockModality("ds", 1.0, 5, 0.2, 2.0),
    mockModality("sd", 4.0, 5, 0.2, 4.0)
  )

  private val multiDistance = new MultiDistance(components)

  test("distance") {
    val dd = 0.5 * (1.4 / 2.0) * (1.4 / 2.0)
    val ss = 0.1 * (1.8 / 3.0) * (1.8 / 3.0)
    val ds = 0.2 * (1.0 / 2.0) * (1.0 / 2.0)
    val sd = 0.2 * (4.0 / 4.0) * (4.0 / 4.0)
    assert(deq(multiDistance(xRow, yRow), sqrt(dd + ss + ds + sd)))
  }

  test("plus") {
    val dd = dense(4.5, 3.0, 2.5, 7.0, 5.5)
    val ss = dense(7.0, 2.0, 3.0, 0.0, 2.5, 0.0, 4.5)
    val ds = dense(3.5, 8.5, 0.0, 8.5, 3.5)
    val sd = dense(5.0, 3.0, 6.5, 1.0, 3.5)
    val expected = new GenericRowWithSchema(Array(dd, ss, ds, sd), schema)
    assert(rowEquals(expected, multiDistance.plus(xRow, yRow, a = 0.5)))
  }

  test("centroid") {
    val dd = dense(2.5, 2.0, 2.0, 4.5, 4.0)
    val ss = dense(3.5, 2.0, 1.5, 0.0, 1.5, 0.0, 4.5)
    val ds = dense(3.5, 5.5, 0.0, 5.5, 3.5)
    val sd = dense(2.5, 1.5, 4.5, 0.5, 2.0)
    val expected = new GenericRowWithSchema(Array(dd, ss, ds, sd), schema)
    assert(rowEquals(expected, multiDistance.centroid(Array(xRow, yRow))))
  }

  test("scale") {
    val dd = dense(0.5, 1.0, 1.5, 2.0, 2.5)
    val ss = sparse(7, Seq(1 -> 2.0, 4 -> 0.5, 6 -> 4.5))
    val ds = dense(3.5, 2.5, 0.0, 2.5, 3.5)
    val sd = sparse(5, Seq(2 -> 2.5, 4 -> 0.5))
    val expected = new GenericRowWithSchema(Array(dd, ss, ds, sd), schema)
    assert(rowEquals(expected, multiDistance.scale(xRow, 0.5)))
  }

  test("aggregation") {
    val dd = dense(1.0, 2.0, 1.0, 4.0, 3.0)
    val ss = sparse(7, Seq(4 -> 1.0))
    val ds = sparse(5, Seq(1 -> 5.0, 3 -> 5.0))
    val sd = sparse(5, Seq(2 -> 4.0, 4 -> 1.0))
    val expected = new GenericRowWithSchema(Array(dd, ss, ds, sd), schema)
    assert(rowEquals(expected, multiDistance.agg(xRow, yRow, min)))
  }

  private def mockModality(modality: String, value: Double, dim: Int, weight: Double, norm: Double): ModalWrapper = {
    val fun = new ModalDistance {
      override def dense(x: DenseVector, y: DenseVector): Double = value

      override def mix(x: DenseVector, y: SparseVector): Double = value

      override def sparse(x: SparseVector, y: SparseVector): Double = value
    }
    new ModalWrapper(fun, modality, dim, weight, norm)
  }

  private def rowEquals(x: Row, y: Row): Boolean = {
    schema.forall({ field =>
      val xv = x.getAs[Vector](field.name)
      val yv = y.getAs[Vector](field.name)
      (xv, yv) match {
        case (dx: DenseVector, dy: DenseVector) => dx == dy
        case (sx: SparseVector, sy: SparseVector) => sx == sy
        case _ => false
      }
    })
  }
}
