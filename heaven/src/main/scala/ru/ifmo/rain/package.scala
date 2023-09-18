package ru.ifmo

import org.apache.spark.SparkContext
import org.apache.spark.sql.types.{IntegerType, LongType, StructField}

import java.util.logging.{ConsoleHandler, Level, Logger, SimpleFormatter}

package object rain {
  val ID_COL: String = "_id"
  val ID_FIELD: StructField = StructField(ID_COL, LongType, nullable = false)

  val LABEL_COL: String = "_label"
  val LABEL_FIELD: StructField = StructField(LABEL_COL, IntegerType, nullable = false)

  val CLASS_COL: String = "_class"

  val NOISE_LABEL: Int = -1

  System.setProperty("java.util.logging.SimpleFormatter.format", "%4$s: %5$s %n")
  private val handler = new ConsoleHandler()
  handler.setFormatter(new SimpleFormatter())

  val logger: Logger = Logger.getLogger("HEAVEN")
  logger.setLevel(Level.WARNING)
  logger.addHandler(handler)

  def enableDev(): Unit = logger.setLevel(Level.INFO)

  def withClearCaches[R](action: => R): R = {
    val cachedBefore = SparkContext.getOrCreate().getPersistentRDDs
    val result = try {
      action
    } finally {
      SparkContext.getOrCreate().getPersistentRDDs
        .filterKeys { !cachedBefore.contains(_) }
        .foreach { _._2.unpersist(blocking = false) }
    }
    result
  }

  def withTime[R](taskName: String, clear: Boolean = false)(action: => R): R = {
    val start = System.currentTimeMillis()
    try {
      val result = if (clear) withClearCaches(action) else action
      val elapsed = (System.currentTimeMillis() - start).toDouble / 1000
      logger.info(s"$taskName: ${elapsed}s")
      result
    } catch {
      case e: Throwable =>
        val elapsed = (System.currentTimeMillis() - start).toDouble / 1000
        logger.info(s"$taskName failed: ${elapsed}s")
        throw e
    }
  }
}
