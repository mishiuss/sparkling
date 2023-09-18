package ru.ifmo.rain.algorithms

import ru.ifmo.rain.algorithms.DBSCANTest.computers
import ru.ifmo.rain.algorithms.dbscan.DBSCAN
import ru.ifmo.rain.algorithms.sandbox.{BlobsSandbox, Sandbox, TestSetup, UniModalSandbox}

import scala.math.{cbrt, sqrt}

trait DBSCANTest extends Sandbox {
  override def forSetup(name: String)(setupSupplier: String => TestSetup): Unit = {
    List(0.02 -> "few", 0.06 -> "few", 0.06 -> "mid", 0.11 -> "mid", 0.11 -> "lot", 0.17 -> "lot")
      .foreach { case (eps, pts) =>
        List(false, true).foreach { borderNoise =>
          runWithName(s"$name {epsilon: $eps, minPoints: $pts, borderNoise: $borderNoise}") {
            val setup = setupSupplier(name)
            val minPoints = computers(pts)(setup.n)
            val pointsInBox = sqrt(5L * setup.n).toLong
            val maxClusters = sqrt(setup.n).toInt

            val algo = new DBSCAN(eps, minPoints, borderNoise, maxClusters, pointsInBox)
            val model = algo.fit(setup.df, setup.dist)

            evaluate(model.dataframe(), setup.dist)
            model.dataframe().unpersist(blocking = false)
          }
        }
      }
  }
}

object DBSCANTest {
  private val computers = Map[String, Long => Long](
    "few" -> (n => cbrt(n / 7L).toLong),
    "mid" -> (n => cbrt(n).toLong),
    "lot" -> (n => cbrt(n * 7L).toLong)
  )
}

class DBSCANUniModalTest extends UniModalSandbox with DBSCANTest

class DBSCANBlobsTest extends BlobsSandbox with DBSCANTest