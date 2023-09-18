package ru.ifmo.rain.measures

import ru.ifmo.rain.Sparkling
import ru.ifmo.rain.measures.ExtConjugacy.comb
import ru.ifmo.rain.utils.Compares.deq

import scala.math.{log, sqrt}


/**
 * Evaluates external measures based on conjugacy matrix
 */
@Sparkling
class ExtConjugacy(val matrix: Array[Array[Long]]) {
  private val matrixT = matrix.transpose
  private val a = matrix.map { _.sum }.zipWithIndex
  private val b = matrixT.map { _.sum }.zipWithIndex
  private val n = a.map { _._1 }.sum

  /**
   * $$ F = \sum_jp_jmax_i [2 \frac{p_{ij}}{p_i} \frac{p_{ij}}{p_j} / (\frac{p_{ij}}{p_i} + \frac{p_{ij}}{p_j})] $$
   * The index should increase.
   */
  @Sparkling
  lazy val f1: Double = b.map({ case (nj, j) =>
    nj / n.toDouble * a.map({ case (ni, i) =>
      val nij = matrix(i)(j).toDouble
      val (precision, recall) = (nij / ni, nij / nj)
      val div = precision + recall
      if (deq(div, 0.0)) 0.0
      else 2.0 * precision * recall / div
    }).max
  }).sum

  /**
   * $$ P = \sum_i max_jp_{ij} $$
   * Limited to the values 0 and 1, where 1 means that the clustering results exactly match the markup.
   */
  @Sparkling
  lazy val purity: Double = matrix.map { row => row.map { _ / n.toDouble }.max }.sum

  /**
   * $$ E = -\sum_i p_i (\sum_i \frac{p_{ij}}{p_i} \log (\frac{p_{ij}}{p_i})) $$
   */
  @Sparkling
  lazy val entropy: Double = -a.map({ case (ni, i) =>
    ni / n.toDouble * matrixT.map { row =>
      val pij = row(i).toDouble / ni
      if (deq(pij, 0.0)) 0.0
      else pij * log(pij)
    }.sum
  }).sum

  /**
   * $$ MS = \dfrac{\sqrt{\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2} - 2 \sum_{ij} \binom{n_{ij}}{2}}}{\sqrt{\sum_j \binom{b_j}{2}}} $$
   */
  @Sparkling
  lazy val minkowski: Double = {
    val aCombSum = a.map { case (ni, _) => comb(ni) }.sum
    val bCombSum = b.map { case (nj, _) => comb(nj) }.sum
    val nCombSum = matrix.map { _.map(comb).sum }.sum
    sqrt(aCombSum + bCombSum - 2L * nCombSum) / sqrt(bCombSum)
  }

  /**
   * $$ ARI = \frac{\sum_{ij} \binom{n_{ij}}{2} - {[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}] / \binom{n}{2}}}{{\frac{1}{2} [\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}]} - {[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}] / \binom{n}{2}}} $$
   * Limited to -1 and 1, where 1 means that the clustering results exactly match the markup.
   */
  @Sparkling
  lazy val adjustedRand: Double = {
    val index = matrix.map { _.map(comb).sum }.sum
    val aCombSum = a.map { case (ni, _) => comb(ni) }.sum
    val bCombSum = b.map { case (nj, _) => comb(nj) }.sum
    val expectedIndex = a.map { case (ni, _) => comb(ni) * bCombSum }.sum / comb(n)
    val maxIndex = (aCombSum + bCombSum) / 2L
    (index - expectedIndex).toDouble / (maxIndex - expectedIndex)
  }

  /**
   * $$ GK = \sum_i p_i (1 - max_j\frac{p_{ij}}{p_i}) $$
   */
  @Sparkling
  lazy val goodmanKruskal: Double = a.map({ case (ni, i) =>
    ni.toDouble / n * (1.0 - matrixT.map { row => row(i).toDouble / ni }.max)
  }).sum

  /**
   * $$ VI = -\sum_i p_i \log p_i - \sum_i p_j \log p_j - 2 \sum_i \sum_j p_{ij} \log \dfrac{p_{ij}}{p_ip_j} $$
   */
  @Sparkling
  lazy val varInformation: Double = {
    val ei = a.map { case (ni, _) => ni.toDouble / n * log(ni.toDouble / n) }.sum
    val ej = b.map { case (nj, _) => nj.toDouble / n * log(nj.toDouble / n) }.sum
    val eij = matrix.zipWithIndex.map({ case(row, i) =>
      row.zipWithIndex.map({ case (nij, j) =>
        val div = a(i)._1 * b(j)._1
        if (deq(div, 0.0) || nij == 0L) 0.0
        else nij.toDouble / n * log(nij.toDouble / div)
      }).sum
    }).sum
    -ei - ej - 2.0 * eij
  }

  override def toString: String =
    s"{f1: $f1, purity: $purity, entropy: $entropy, " +
      s"minkowski: $minkowski, adjustedRand:$adjustedRand, " +
      s"goodmanKruskal: $goodmanKruskal, varInformation: $varInformation}"
}

object ExtConjugacy {
  def comb(x: Long): Long = x * (x - 1L) / 2L
}
