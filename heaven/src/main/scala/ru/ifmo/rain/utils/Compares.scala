package ru.ifmo.rain.utils

import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.ml.linalg.Vector

/**
 * Comparison utils for multimodal objects
 */
object Compares {
  def modalities(obj: Row): Array[String] =
    obj.asInstanceOf[GenericRowWithSchema].schema
      .filterNot(_.name.startsWith("_")).map(_.name)
      .sorted.toArray

  def compare(x: Row, y: Row): Int = {
    val xObj = x.asInstanceOf[GenericRowWithSchema]
    val xModalities = modalities(x)

    val yObj = y.asInstanceOf[GenericRowWithSchema]
    val yModalities = modalities(y)

    if (!xModalities.sameElements(yModalities))
      throw new IllegalArgumentException("Objects are not of the same modalities")

    xModalities.foreach { modality =>
      val xModal = xObj.getAs[Vector](modality)
      val yModal = yObj.getAs[Vector](modality)
      if (xModal.size != yModal.size)
        throw new IllegalArgumentException("Objects are not of the same length")
      for (idx <- 0 until xModal.size) {
        val (xv, yv) = (xModal(idx), yModal(idx))
        if (!deq(xv, yv)) return xv compareTo yv
      }
    }
    0
  }

  def deq(x: Double, y: Double): Boolean = math.abs(x - y) < 1e-12

  def dlt(x: Double, y: Double): Boolean = x < y || deq(x, y)

  def dgt(x: Double, y: Double): Boolean = x > y || deq(x, y)

  def equals(x: Row, y: Row): Boolean = try {
    compare(x, y) == 0
  } catch {
    case _: Exception => false
  }
}
