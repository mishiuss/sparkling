package ru.ifmo.rain.algorithms.dbscan.box

import ru.ifmo.rain.utils.Compares.{dgt, dlt}

import scala.math.min


class DimBound(val lower: Double, val upper: Double, val inclusive: Boolean = false) extends Serializable {

  def isNumberWithin(n: Double): Boolean = dgt(n, lower) && ((n < upper) || (inclusive && dlt(n, upper)))

  def split(n: Int, minLen: Double): List[DimBound] = {
    val maxN = ((this.width / minLen) + 0.5).toInt
    split(min(n, maxN))
  }

  def split(n: Int): List[DimBound] = {
    var result: List[DimBound] = Nil
    val increment = (upper - lower) / n
    var currentLower = lower

    for (i <- 1 to n) {
      val include = if (i < n) false else this.inclusive
      val newUpper = currentLower + increment
      val newSplit = new DimBound(currentLower, newUpper, include)
      result = newSplit :: result
      currentLower = newUpper
    }

    result.reverse
  }

  def width: Double = upper - lower

  def extend(byLength: Double): DimBound = {
    val halfLength = byLength / 2
    new DimBound(this.lower - halfLength, this.upper + halfLength, this.inclusive)
  }

  def extend(by: DimBound): DimBound = extend(by.width)

  override def toString: String = "[" + lower + " - " + upper + (if (inclusive) "]" else ")")

  override def equals(that: Any): Boolean = that match {
    case typedThat: DimBound =>
      typedThat.canEqual(this) &&
        this.lower == typedThat.lower &&
        this.upper == typedThat.upper &&
        this.inclusive == typedThat.inclusive
    case _ =>
      false
  }

  override def hashCode(): Int = 41 * (41 * (41 + (if (inclusive) 1 else 0)) + lower.toInt) + upper.toInt

  def canEqual(other: Any): Boolean = other.isInstanceOf[DimBound]
}