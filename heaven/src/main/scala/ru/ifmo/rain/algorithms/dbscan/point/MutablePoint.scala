package ru.ifmo.rain.algorithms.dbscan.point


private[dbscan] class MutablePoint(p: Point, val tempId: Int) extends Point(p) {
  var transientClusterId: Long = p.clusterId
  var visited: Boolean = false

  def toImmutablePoint: Point = new Point(
    obj, pointId, boxId, distanceFromOrigin, neighboursCount, transientClusterId
  )
}
