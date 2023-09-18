package ru.ifmo.rain.distances

import ru.ifmo.rain.utils.Compares.deq


class CosineTest extends ModalDistanceTest {

  test("cosine dense to dense") {
    assert(deq(Cosine(dx, dy), 0.1636363636363637))
  }

  test("cosine sparse to sparse") {
    assert(deq(Cosine(sx, sy), 0.9743420997104607))
  }

  test("cosine dense to sparse") {
    assert(deq(Cosine(dw, sw), 0.41876180628090365))
  }

  test("cosine sparse to dense") {
    assert(deq(Cosine(dz, sz), 0.4176748639268918))
  }

}
