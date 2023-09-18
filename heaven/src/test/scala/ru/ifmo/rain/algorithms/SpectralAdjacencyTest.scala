package ru.ifmo.rain.algorithms

import ru.ifmo.rain.algorithms.sandbox.{BlobsSandbox, Sandbox, TestSetup, UniModalSandbox}
import ru.ifmo.rain.algorithms.spectral.SpectralAdjacency

import scala.math.sqrt


trait SpectralAdjacencyTest extends Sandbox {
  override def forSetup(name: String)(setupSupplier: String => TestSetup): Unit = {
    List(7, 13, 19).foreach { eigens =>
      List(5, 11, 20).foreach { neighbours =>
        List(2, 5, 9).foreach { k =>
          runWithName(s"$name {eigens: $eigens, neighbours: $neighbours, k: $k}") {
            val setup = setupSupplier(name)
            val maxIterations = sqrt(2L * setup.n).toInt

            val algo = new SpectralAdjacency(neighbours, eigens, k, maxIterations, 42L)
            val model = algo.fit(setup.df, setup.dist)

            evaluate(model.dataframe(), setup.dist)
            model.dataframe().unpersist(blocking = false)
          }
        }
      }
    }
  }
}

class SpectralAdjacencyUniModalTest extends UniModalSandbox with SpectralAdjacencyTest

class SpectralAdjacencyBlobsTest extends BlobsSandbox with SpectralAdjacencyTest
