package ru.ifmo.rain.distances

import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import ru.ifmo.rain.utils.BLAS.l2
import ru.ifmo.rain.utils.Compares.deq

/**
 * Implementation of cosine distance
 * $$ cosine(x, y) = \frac{x * y}{||x||_2 * ||y||_2} $$
 * $$ x * y = x_1 * y_1 + x_2 * y_2 + \dots + x_n * y_n $$
 * $$ |||z||_2 = \sqrt{z_1^2 + z_2^2 + \dots + z_n^2} $$
 */
object Cosine extends ModalDistance {

  override def dense(x: DenseVector, y: DenseVector): Double = {
    val xy = l2(x) * l2(y)
    if (deq(xy, 0.0)) 0.0 else 1.0 - dot(x, y) / xy
  }

  override def mix(x: DenseVector, y: SparseVector): Double = {
    val xy = l2(x) * l2(y)
    if (deq(xy, 0.0)) 0.0 else 1.0 - dot(x, y) / xy
  }

  override def sparse(x: SparseVector, y: SparseVector): Double = {
    val xy = l2(x) * l2(y)
    if (deq(xy, 0.0)) 0.0 else 1.0 - dot(x, y) / xy
  }

  def dot(x: DenseVector, y: DenseVector): Double = {
    var result = 0.0
    for (idx <- 0 until x.size) result += x(idx) * y(idx)
    result
  }

  def dot(x: DenseVector, y: SparseVector): Double = {
    var result = 0.0
    y.foreachActive((idx, value) => result += x(idx) * value)
    result
  }

  def dot(x: SparseVector, y: SparseVector): Double = {
    val (xv, yv) = (x.values, y.values)
    val (xi, yi) = (x.indices, y.indices)
    val (xl, yl) = (xv.length, yv.length)

    var (kx, ky) = (0, 0)
    var result = 0.0
    while (kx < xl && ky < yl) {
      val ix = xi(kx)
      while (ky < yl && yi(ky) < ix) ky += 1
      if (ky < yl && yi(ky) == ix) {
        result += xv(kx) * yv(ky)
        ky += 1
      }
      kx += 1
    }
    result
  }
}
