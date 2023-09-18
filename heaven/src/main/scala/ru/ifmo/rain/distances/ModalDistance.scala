package ru.ifmo.rain.distances

import org.apache.spark.ml.linalg.{Vector, DenseVector, SparseVector}


/**
 * Base class for intra-modal distances,
 * which can handle both sparse and dense vector representations
 */
abstract class ModalDistance extends Serializable {
  def apply(x: Vector, y: Vector): Double = (x, y) match {
    case (dx: DenseVector, dy: DenseVector) => dense(dx, dy)
    case (dx: DenseVector, sy: SparseVector) => mix(dx, sy)
    case (sx: SparseVector, dy: DenseVector) => mix(dy, sx)
    case (sx: SparseVector, sy: SparseVector) => sparse(sx, sy)
  }

  def dense(x: DenseVector, y: DenseVector): Double

  def mix(x: DenseVector, y: SparseVector): Double

  def sparse(x: SparseVector, y: SparseVector): Double
}