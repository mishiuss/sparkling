package ru.ifmo.rain.algorithms

import ru.ifmo.rain.algorithms.sandbox.{BlobsSandbox, Sandbox, TestSetup, UniModalSandbox}
import ru.ifmo.rain.algorithms.spectral.SpectralSimilarity

import scala.math.sqrt

trait SpectralSimilarityTest extends Sandbox {
  override def forSetup(name: String)(setupSupplier: String => TestSetup): Unit = {
    List(7, 13, 19).foreach { eigens =>
      List(0.1, 0.4, 1.0).foreach { gamma =>
        List(2, 5, 9).foreach { k =>
          runWithName(s"$name {eigens: $eigens, gamma: $gamma, k: $k}") {
            val setup = setupSupplier(name)
            val maxIterations = sqrt(2L * setup.n).toInt

            val algo = new SpectralSimilarity(gamma, eigens, k, maxIterations, 42L)
            val model = algo.fit(setup.df, setup.dist)

            evaluate(model.dataframe(), setup.dist)
            model.dataframe().unpersist(blocking = false)
          }
        }
      }
    }
  }
}

class SpectralSimilarityUniModalTest extends UniModalSandbox with SpectralSimilarityTest

class SpectralSimilarityBlobsTest extends BlobsSandbox with SpectralSimilarityTest
