package ru.ifmo.rain.distances

import org.apache.spark.ml.linalg.Vector

/**
 * Holds meta information for a single modality
 * @param fun Inta-modal distance metric
 * @param modality Unique string identifier
 * @param dim Dimension of vector representation
 * @param weight Modality's "importance"
 * @param norm Normalisation coefficient
 */
class ModalWrapper(
                    val fun: ModalDistance,
                    val modality: String,
                    val dim: Int,
                    val weight: Double,
                    val norm: Double
                  ) extends Serializable {

  /**
   * Normalised intra-modal distance between vector representations
   * @param x Modality's vector representation of first multimodal object
   * @param y Modality's vector representation of second multimodal object
   * @return Dissimilarity value in range [0, 1]
   */
  def apply(x: Vector, y: Vector): Double = fun(x, y) / norm
}
