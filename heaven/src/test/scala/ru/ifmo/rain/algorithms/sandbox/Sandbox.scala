package ru.ifmo.rain.algorithms.sandbox

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.funsuite.AnyFunSuite
import ru.ifmo.rain._
import ru.ifmo.rain.distances.MultiDistance
import ru.ifmo.rain.measures.{Externals, Internals}

class TestSetup(val df: DataFrame, val n: Long, val dist: MultiDistance)

trait Sandbox extends AnyFunSuite {
  def forSetup(name: String)(setupSupplier: String => TestSetup): Unit

  def evaluate(df: DataFrame, dist: MultiDistance): Unit = {
    val withoutNoise = df.filter(df.col(LABEL_COL) =!= NOISE_LABEL)
    val noiseRatio = (100 * withoutNoise.count() / df.count().toDouble).toInt
    if (noiseRatio < 90)
      throw new IllegalStateException(s"Too much noise: only $noiseRatio% of objects are labeled")

    val labelsClasses = withoutNoise.select(ID_COL, LABEL_COL, CLASS_COL)
    elapse("Externals") { Externals.pairwise(labelsClasses) }
    elapse("Conjugate") { Externals.conjugate(labelsClasses) }

    for (measure <- Array("ch", "db", "gd43", "sil*"))
      elapse(measure) { Internals.evaluate(measure, withoutNoise, dist)}
  }

  protected def elapse[R](taskName: String)(action: => R): Unit = {
    val start = System.currentTimeMillis()
    try {
      val result = action
      val elapsed = (System.currentTimeMillis() - start).toDouble / 1000
      logger.info(s"$taskName = $result, ${elapsed}s")
    } catch {
      case _: Throwable =>
        val elapsed = (System.currentTimeMillis() - start).toDouble / 1000
        logger.warning(s"$taskName failed: ${elapsed}s")
    }
  }

  protected def runWithName(name: String)(suiteFun: => Unit): Unit = {
    enableDev()
    val fullName = s"${getClass.getSimpleName}: $name"
    test(fullName) {
      logger.info(s"--- $fullName ---")
      try {
        suiteFun
      } catch {
        case e: IllegalStateException =>
          logger.warning(e.getMessage)
      }
    }
  }
}

object Sandbox {
  val ss: SparkSession = SparkSession.builder()
    .master("local[*]")
    .appName("sandbox")
    .getOrCreate()
  ss.sparkContext.setLogLevel("WARN")
  ss.sparkContext.setCheckpointDir("src/test/checkpoint")
}