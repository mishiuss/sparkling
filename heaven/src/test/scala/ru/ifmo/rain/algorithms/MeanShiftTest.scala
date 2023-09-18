package ru.ifmo.rain.algorithms

import org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK
import ru.ifmo.rain.algorithms.meanshift.MeanShift
import ru.ifmo.rain.algorithms.sandbox.{BlobsSandbox, Sandbox, TestSetup, UniModalSandbox}
import ru.ifmo.rain.logger

import scala.math.sqrt


trait MeanShiftTest extends Sandbox {
  override def forSetup(name: String)(setupSupplier: String => TestSetup): Unit = {
    List(0.03, 0.07, 0.12, 0.18, 0.25).foreach { radius =>
      runWithName(s"$name {radius: $radius}") {
        val setup = setupSupplier(name)
        val maxClusters = sqrt(setup.n).toInt
        val maxIterations = sqrt(2L * setup.n).toInt
        val initial = maxIterations

        val algo = new MeanShift(radius, maxClusters, maxIterations, initial)
        val model = algo.fit(setup.df, setup.dist)

        logger.info("- Without noise detection -")
        val withoutNoise = model.setNoise(false)
          .predict(setup.df).persist(MEMORY_AND_DISK)
        evaluate(withoutNoise, setup.dist)
        withoutNoise.unpersist(blocking = false)

        logger.info("- With noise detection -")
        val withNoise = model.setNoise(true)
          .predict(setup.df).persist(MEMORY_AND_DISK)
        evaluate(withNoise, setup.dist)
        withNoise.unpersist(blocking = false)
      }
    }
  }
}

class MeanShiftUniModalTest extends UniModalSandbox with MeanShiftTest

class MeanShiftBlobsTest extends BlobsSandbox with MeanShiftTest
