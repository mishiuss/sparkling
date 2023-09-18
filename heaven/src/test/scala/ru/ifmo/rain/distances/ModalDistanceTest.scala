package ru.ifmo.rain.distances

import org.apache.spark.ml.linalg.Vectors.{dense, sparse}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.scalatest.funsuite.AnyFunSuite


class ModalDistanceTest extends AnyFunSuite {
  val dx: DenseVector = dense(1.0, 2.0, 3.0, 4.0, 5.0).asInstanceOf[DenseVector]
  val dy: DenseVector = dense(4.0, 2.0, 1.0, 5.0, 3.0).asInstanceOf[DenseVector]

  val sx: SparseVector = sparse(7, Seq((1, 4.0), (4, 1.0), (6, 9.0))).asInstanceOf[SparseVector]
  val sy: SparseVector = sparse(7, Seq((0, 7.0), (2, 3.0), (4, 2.0))).asInstanceOf[SparseVector]

  val dw: DenseVector = dense(7.0, 5.0, 0.0, 5.0, 7.0).asInstanceOf[DenseVector]
  val sw: SparseVector = sparse(5, Seq((1, 6.0), (3, 6.0))).asInstanceOf[SparseVector]

  val sz: SparseVector = sparse(5, Seq((2, 5.0), (4, 1.0))).asInstanceOf[SparseVector]
  val dz: DenseVector = dense(5.0, 3.0, 4.0, 1.0, 3.0).asInstanceOf[DenseVector]
}
