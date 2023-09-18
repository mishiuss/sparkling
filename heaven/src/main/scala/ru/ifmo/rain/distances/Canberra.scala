package ru.ifmo.rain.distances

import org.apache.spark.ml.linalg.{DenseVector, SparseVector}

import scala.math.abs

/**
 * Implementation of canberra distance
 * $$ canberra(x, y) = \frac{|x_1 - y_1|}{|x_1| + |y_1|} + \frac{|x_2 - y_2|}{|x_2| + |y_2|} + \dots + \frac{|x_n - y_n|}{|x_n| + |y_n|} $$
 */
object Canberra extends ModalDistance {
  override def dense(x: DenseVector, y: DenseVector): Double = {
    var result = 0.0
    for (idx <- 0 until x.size) {
      val diff = abs(x(idx) - y(idx))
      val sum = abs(x(idx)) + abs(y(idx))
      if (sum > 1e-9) result += diff / sum
    }
    result
  }

  override def mix(x: DenseVector, y: SparseVector): Double = {
    val (yv, yi) = (y.values, y.indices)
    val yl = yv.length
    var (ky, result) = (0, 0.0)
    for (ix <- 0 until x.size) {
      if (ky < yl && yi(ky) == ix) {
        val diff = abs(x(ix) - yv(ky))
        val sum = abs(x(ix)) + abs(yv(ky))
        if (sum > 1e-9) result += diff / sum
        ky += 1
      }
      else if (abs(x(ix)) > 1e-9) {
        result += 1.0
      }
    }
    result
  }

  override def sparse(x: SparseVector, y: SparseVector): Double = {
    if (x.size == 0 && y.size == 0) return 0.0
    val (xt, yt) = if (y.size > x.size) (y, x) else (x, y)

    val (xv, yv) = (xt.values, yt.values)
    val (xi, yi) = (xt.indices, yt.indices)
    val (xl, yl) = (xv.length, yv.length)

    var (kx, ky) = (0, 0)
    var result = 0.0
    while (kx < xl) {
      val ix = xi(kx)
      while (ky < yl && yi(ky) < ix) {
        if (abs(yv(ky)) > 1e-9) result += 1.0
        ky += 1
      }
      if (ky < yl && yi(ky) == ix) {
        val diff = abs(xv(kx) - yv(ky))
        val sum = abs(xv(kx)) + abs(yv(ky))
        if (sum > 1e-9) result += diff / sum
        ky += 1
      }
      else if (abs(xv(kx)) > 1e-9) {
        result += 1.0
      }
      kx += 1
    }
    result
  }
}