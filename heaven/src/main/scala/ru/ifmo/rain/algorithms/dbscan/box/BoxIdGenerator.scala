package ru.ifmo.rain.algorithms.dbscan.box

private [dbscan] class BoxIdGenerator(val initialId: Int) {
  var nextId: Int = initialId

  def getNextId: Int = { nextId += 1; nextId }
}
