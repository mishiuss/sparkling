package ru.ifmo.rain.measures

import ru.ifmo.rain.Sparkling

import scala.math.sqrt


/**
 * Evaluates external measures based on TP, FP, FN, TN values
 */
@Sparkling
class ExtPairwise(val tp: Double, val fp: Double, val fn: Double, val tn: Double) extends Serializable {

  /**
   * $$ Rand = \frac{TP + TN}{TP + TN + FP + FN} $$
   * Limited to the values 0 and 1, where 1 means that the clustering results exactly match the markup.
   */
  @Sparkling
  lazy val rand: Double = (tp + tn) / (tp + fp + fn + tn)

  /**
   * $$ Jaccard = \frac{TP}{TP + FN + FP} $$
   * Limited to the values 0 and 1, where 1 means that the clustering results exactly match the markup.
   */
  @Sparkling
  lazy val jaccard: Double = tp / (tp + fp + fn)

  /**
   * $$ FM = \sqrt{\frac{TP}{TP + FP} \cdot \frac{TP}{TP + FN}} $$
   * The index should increase.
   */
  @Sparkling
  lazy val fowlkesMallows: Double = sqrt(tp / (tp + fp) * tp / (tp + fn))

  /**
   * $$ \Phi = \frac{TP \times TN - FN \times FP}{(TP + FN)(TP + FP)(FN + TN)(FP + TN)} $$
   * The index should increase.
   */
  @Sparkling
  lazy val phi: Double = (tp * tn - fp * fn) / ((tp + fn) * (tp + fp) * (fn + tn) * (fp + tn))

  override def toString = s"{rand: $rand, jaccard: $jaccard, fowlkesMallows: $fowlkesMallows, phi: $phi}"
}