package ru.ifmo.rain.distances

import ru.ifmo.rain.utils.Compares.deq


class CanberraTest extends ModalDistanceTest {

  test("canberra dense to dense") {
    assert(deq(Canberra(dx, dy), 1.4611111111111112))
  }

  test("canberra sparse to sparse") {
    assert(deq(Canberra(sx, sy), 4.333333333333334))
  }

  test("canberra dense to sparse") {
    assert(deq(Canberra(dw, sw), 2.1818181818181817))
  }

  test("canberra sparse to dense") {
    assert(deq(Canberra(dz, sz), 3.6111111111111118))
  }
}
