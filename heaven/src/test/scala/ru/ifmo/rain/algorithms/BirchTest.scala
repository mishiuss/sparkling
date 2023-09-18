package ru.ifmo.rain.algorithms

import org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK
import ru.ifmo.rain.algorithms.birch.Birch
import ru.ifmo.rain.algorithms.sandbox.{BlobsSandbox, Sandbox, TestSetup, UniModalSandbox}

trait BirchTest extends Sandbox {
  override def forSetup(name: String)(setupSupplier: String => TestSetup): Unit = {
    List(5, 12, 25).foreach { maxBranches =>
      List(0.1, 0.3, 0.7).foreach { threshold =>
        List(2, 5, 14).foreach { k =>
          runWithName(s"$name {maxBranches: $maxBranches, threshold: $threshold, k: $k}") {
            val setup = setupSupplier(name)
            val maxIterations = math.sqrt(setup.n).toInt

            val algo = new Birch(k, maxBranches, threshold, maxIterations)
            val model = algo.fit(setup.df, setup.dist)

            val predictions = model.predict(setup.df).persist(MEMORY_AND_DISK)
            evaluate(predictions, setup.dist)
            predictions.unpersist(blocking = false)
          }
        }
      }
    }
  }
}

class BirchUniModalTest extends UniModalSandbox with BirchTest

class BirchBlobsTest extends BlobsSandbox with BirchTest