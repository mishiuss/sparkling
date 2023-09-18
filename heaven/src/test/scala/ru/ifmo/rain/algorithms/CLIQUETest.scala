package ru.ifmo.rain.algorithms

import org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK
import ru.ifmo.rain.algorithms.clique.CLIQUE
import ru.ifmo.rain.algorithms.sandbox.{BlobsSandbox, Sandbox, TestSetup, UniModalSandbox}


trait CLIQUETest extends Sandbox {
  override def forSetup(name: String)(setupSupplier: String => TestSetup): Unit = {
    List(0.05, 0.12, 0.25).foreach { threshold =>
      List(3, 7, 11).foreach { splits =>
        List(3, 6, 10).foreach { levels =>
          runWithName(s"$name {threshold: $threshold, splits: $splits, levels: $levels}") {
            val setup = setupSupplier(name)

            val algo = new CLIQUE(threshold, splits, levels)
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

class CLIQUEUniModalTest extends UniModalSandbox with CLIQUETest

class CLIQUEBlobsTest extends BlobsSandbox with CLIQUETest

