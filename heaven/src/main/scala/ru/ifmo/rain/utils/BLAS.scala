package ru.ifmo.rain.utils

import org.apache.spark.ml.linalg._
import ru.ifmo.rain.utils.Compares.deq

import scala.collection.mutable

/**
 * Implements set of functions for basic linear algebra operations for both dense and sparse vectors
 */
object BLAS {
  def plus(a: Double, x: Vector, b: Double, y: Vector): Vector = (x, y) match {
    case (dx: DenseVector, dy: DenseVector) => BLAS.plus(a, dx, b, dy)
    case (dx: DenseVector, sy: SparseVector) => BLAS.plus(a, dx, b, sy)
    case (sx: SparseVector, dy: DenseVector) => BLAS.plus(a, sx, b, dy)
    case (sx: SparseVector, sy: SparseVector) => BLAS.plus(a, sx, b, sy)
  }

  private def plus(a: Double, dx: DenseVector, b: Double, dy: DenseVector): Vector = {
    val arr = new Array[Double](dx.size)
    for (idx <- arr.indices) arr(idx) = a * dx(idx) + b * dy(idx)
    new DenseVector(arr)
  }

  private def plus(a: Double, dx: DenseVector, b: Double, sy: SparseVector): Vector = {
    val arr = new Array[Double](dx.size)
    val (yv, yi) = (sy.values, sy.indices)
    val yl = yv.length
    var ky = 0
    for (ix <- 0 until dx.size) {
      if (ky < yl && yi(ky) == ix) {
        arr(ix) = a * dx(ix) + b * yv(ky)
        ky += 1
      }
      else {
        arr(ix) = a * dx(ix)
      }
    }
    new DenseVector(arr)
  }
  
  private def plus(a: Double, sx: SparseVector, b: Double, dy: DenseVector): Vector = {
    val arr = new Array[Double](dy.size)
    val (xv, xi) = (sx.values, sx.indices)
    val xl = xv.length
    var kx = 0
    for (iy <- 0 until dy.size) {
      if (kx < xl && xi(kx) == iy) {
        arr(iy) = a * xv(kx) + b * dy(iy)
        kx += 1
      }
      else {
        arr(iy) = b * dy(iy)
      }
    }
    new DenseVector(arr)
  }

  private def plus(a: Double, sx: SparseVector, b: Double, sy: SparseVector): Vector = {
    val (xt, yt) = if (sy.size > sx.size) (sy, sx) else (sx, sy)

    val (xv, yv) = (xt.values, yt.values)
    val (xi, yi) = (xt.indices, yt.indices)
    val (xl, yl) = (xv.length, yv.length)

    val indices = new mutable.MutableList[Int]
    val values = new mutable.MutableList[Double]

    var (kx, ky) = (0, 0)
    while (kx < xl) {
      val ix = xi(kx)
      while (ky < yl && yi(ky) < ix) {
        indices += yi(ky)
        values += b * yv(ky)
        ky += 1
      }
      if (ky < yl && yi(ky) == ix) {
        indices += ix
        values += a * xv(kx) + b * yv(ky)
        ky += 1
      }
      else {
        indices += ix
        values += a * xv(kx)
      }
      kx += 1
    }

    if (values.size > sx.size * SPARSITY) {
      val buffer = Array.fill[Double](sx.size)(0.0)
      for (idx <- values.indices) buffer(indices(idx)) = values(idx)
      new DenseVector(buffer)
    }
    else new SparseVector(sx.size, indices.toArray, values.toArray)
  }

  def aggregate(x: Vector, y: Vector, f: (Double, Double) => Double): Vector = {
    var zeros = 0
    val agg = for (d <- 0 until x.size) yield {
      val res = f(x(d), y(d))
      if (deq(res, 0.0)) zeros += 1
      res
    }
    if (zeros <= x.size * SPARSITY) {
      Vectors.dense(agg.toArray)
    }
    else {
      val values = Iterable.tabulate(x.size) { idx => idx -> agg(idx) }
        .filter { case (_, value) => !deq(value, 0.0) }
      Vectors.sparse(x.size, values.toArray)
    }
  }

  def scale(x: Vector, a: Double): Vector = x match {
    case dx: DenseVector => Vectors.dense(dx.values.map(_ * a))
    case sx: SparseVector => Vectors.sparse(sx.size, sx.indices, sx.values.map(_ * a))
  }

  def l2(x: Vector): Double = math.sqrt(x match {
    case dx: DenseVector => dx.values.fold(0.0) { (acc, v) => acc + v * v }
    case sx: SparseVector => sx.values.fold(0.0) { (acc, v) => acc + v * v }
  })

  private val SPARSITY = 0.4
}
