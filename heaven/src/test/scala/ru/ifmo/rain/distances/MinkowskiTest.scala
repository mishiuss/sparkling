package ru.ifmo.rain.distances

import ru.ifmo.rain.distances.Minkowski.{Chebyshev, Euclidean, Manhattan}
import ru.ifmo.rain.utils.Compares.deq

import scala.math.sqrt


class MinkowskiTest extends ModalDistanceTest {

  test("minkowski dense to dense") {
    assert(deq(Manhattan(dx, dy), 8.0))
    assert(deq(Euclidean(dx, dy), sqrt(18.0)))
    assert(deq(Chebyshev(dx, dy), 3.0))
  }

  test("minkowski sparse to sparse") {
    assert(deq(Manhattan(sx, sy), 24.0))
    assert(deq(Euclidean(sx, sy), sqrt(156.0)))
    assert(deq(Chebyshev(sx, sy), 9.0))
  }

  test("minkowski dense to sparse") {
    assert(deq(Manhattan(dw, sw), 16.0))
    assert(deq(Euclidean(dw, sw), sqrt(100.0)))
    assert(deq(Chebyshev(dw, sw), 7.0))
  }

  test("minkowski sparse to dense") {
    assert(deq(Manhattan(dz, sz), 12.0))
    assert(deq(Euclidean(dz, sz), sqrt(40.0)))
    assert(deq(Chebyshev(dz, sz), 5.0))
  }
}
