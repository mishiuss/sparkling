package ru.ifmo.rain.distances

import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import ru.ifmo.rain.distances.Minkowski.{ACCUMULATORS, INVERSES}

import scala.math.{abs, max, sqrt}


/**
 * Implementation of Minkowski distance
 * $$ minkowski_p(x, y) = (|x_1 - y_1|^p + |x_2 - y_2|^p + \dots + |x_n - y_n|^p)^\frac{1}{p} $$
 * @param p Parameter p from formula above
 */
class Minkowski(p: Int) extends ModalDistance {
  private val accumulate = ACCUMULATORS(p)
  private val inverse = INVERSES(p)

  override def dense(x: DenseVector, y: DenseVector): Double = {
    var result = 0.0
    for (idx <- 0 until x.size) {
      val diff = abs(x(idx) - y(idx))
      result = accumulate(result, diff)
    }
    inverse(result)
  }

  override def mix(x: DenseVector, y: SparseVector): Double = {
    val (yv, yi) = (y.values, y.indices)
    val yl = yv.length
    var (ky, result) = (0, 0.0)
    for (ix <- 0 until x.size) {
      if (ky < yl && yi(ky) == ix) {
        val diff = abs(x(ix) - yv(ky))
        result = accumulate(result, diff)
        ky += 1
      }
      else {
        result = accumulate(result, abs(x(ix)))
      }
    }
    inverse(result)
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
        result = accumulate(result, abs(yv(ky)))
        ky += 1
      }
      if (ky < yl && yi(ky) == ix) {
        val diff = abs(xv(kx) - yv(ky))
        result = accumulate(result, diff)
        ky += 1
      }
      else {
        result = accumulate(result, abs(xv(kx)))
      }
      kx += 1
    }
    inverse(result)
  }
}

/**
 * Implementation of Minkowski distance
 */
object Minkowski {
  private val ACCUMULATORS: Map[Int, (Double, Double) => Double] = Map(
    1 -> ((acc, x) => acc + x),
    2 -> ((acc, x) => acc + x * x),
    Int.MaxValue -> ((acc, x) => max(acc, x))
  )
  private val INVERSES: Map[Int, Double => Double] = Map(
    1 -> (x => x),
    2 -> (x => sqrt(x)),
    Int.MaxValue -> (x => x)
  )

  /**
   * Implementation of Manhattan distance, which is Minkowski distance with p = 1
   */
  object Manhattan extends Minkowski(1)

  /**
   * Implementation of Euclidean distance, which is Minkowski distance with p = 2
   */
  object Euclidean extends Minkowski(2)

  /**
   * Implementation of Chebyshev distance, which is Minkowski distance with p = inf
   */
  object Chebyshev extends Minkowski(Int.MaxValue)
}