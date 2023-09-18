package ru.ifmo.rain.algorithms

import org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK
import ru.ifmo.rain.algorithms.bisecting.BisectingKMeans
import ru.ifmo.rain.algorithms.sandbox.{BlobsSandbox, Sandbox, TestSetup, UniModalSandbox}

trait BisectingKMeansTest extends Sandbox {
  override def forSetup(name: String)(setupSupplier: String => TestSetup): Unit = {
    List(2, 3, 5, 7, 11).foreach { k =>
      List(0.2, 0.5, 1.0).foreach { minClusterSize =>
        runWithName(s"$name {k: $k, minClusterSize: $minClusterSize}") {
          val setup = setupSupplier(name)
          val maxIterations = math.sqrt(2L * setup.n).toInt

          val algo = new BisectingKMeans(k, maxIterations, minClusterSize)
          val model = algo.fit(setup.df, setup.dist)

          val predictions = model.predict(setup.df).persist(MEMORY_AND_DISK)
          evaluate(predictions, setup.dist)
          predictions.unpersist(blocking = false)
        }
      }
    }
  }
}

class BisectingKMeansUniModalTest extends UniModalSandbox with BisectingKMeansTest

class BisectingKMeansBlobsTest extends BlobsSandbox with BisectingKMeansTest
