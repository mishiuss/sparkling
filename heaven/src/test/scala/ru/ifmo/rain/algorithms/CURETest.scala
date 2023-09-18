package ru.ifmo.rain.algorithms

import org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK
import ru.ifmo.rain.algorithms.CURETest.representativesMap
import ru.ifmo.rain.algorithms.cure.CURE
import ru.ifmo.rain.algorithms.sandbox.{BlobsSandbox, Sandbox, TestSetup, UniModalSandbox}


trait CURETest extends Sandbox {
  override def forSetup(name: String)(setupSupplier: String => TestSetup): Unit = {
    List(2, 5, 11).foreach { k =>
      List("cbrt", "sqrt").foreach { repr =>
        List(0.1, 0.3, 0.9).foreach { shrinkFactor =>
          List(true, false).foreach { removeOutliers =>
            runWithName(s"$name {k: $k, representatives: $repr, shrinkFactor: $shrinkFactor, removeOutliers: $removeOutliers}") {
              val setup = setupSupplier(name)
              val representatives = representativesMap(repr)(setup.n)

              val algo = new CURE(k, representatives, shrinkFactor, removeOutliers)
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
}

class CUREUniModalTest extends UniModalSandbox with CURETest

class CUREBlobsTest extends BlobsSandbox with CURETest

object CURETest {
  private val representativesMap: Map[String, Long => Int] = Map(
    "cbrt" -> (n => math.cbrt(n).toInt),
    "sqrt" -> (n => math.sqrt(n).toInt)
  )
}