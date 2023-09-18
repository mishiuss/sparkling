package ru.ifmo.rain.algorithms

import org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK
import ru.ifmo.rain.algorithms.kmeans.KMeans
import ru.ifmo.rain.algorithms.sandbox.{BlobsSandbox, Sandbox, TestSetup, UniModalSandbox}

import scala.math.sqrt

trait KMeansTest extends Sandbox {
  override def forSetup(name: String)(setupSupplier: String => TestSetup): Unit = {
    List(2, 3, 5, 7, 11).foreach { k =>
      runWithName(s"$name {k: $k}") {
        val setup = setupSupplier(name)
        val maxIterations = sqrt(2L * setup.n).toInt

        val algo = new KMeans(k, maxIterations, initSteps = 2)
        val model = algo.fit(setup.df, setup.dist)

        val predictions = model.predict(setup.df).persist(MEMORY_AND_DISK)
        evaluate(predictions, setup.dist)
        predictions.unpersist(blocking = false)
      }
    }
  }
}

class KMeansUniModalTest extends UniModalSandbox with KMeansTest

class KMeansBlobsTest extends BlobsSandbox with KMeansTest
