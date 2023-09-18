package ru.ifmo.rain.algorithms.cure

import org.apache.spark.sql.Row
import ru.ifmo.rain.distances.MultiDistance
import ru.ifmo.rain.utils.Compares

import scala.annotation.tailrec


case class CURECluster(
                        points: Array[KDPoint],
                        var representatives: Array[KDPoint],
                        var nearest: CURECluster,
                        var mean: KDPoint,
                        var distance: Double = 0.0d,
                        var id: Int = 0
                      )


case class KDPoint(
                    var obj: Row,
                    var cluster: CURECluster = null
                  ) {

  override def equals(other: scala.Any): Boolean = other match {
    case point: KDPoint => Compares.equals(obj, point.obj)
    case _ => false
  }
}


case class KDNode(
                   point: KDPoint,
                   var left: KDNode,
                   var right: KDNode,
                   var deleted: Boolean = false
                 )


case class KDTree(var root: KDNode, dist: MultiDistance) {

  private def newNode(point: KDPoint): KDNode = KDNode(point, null, null)

  def insert(point: KDPoint): KDNode = insertRec(this.root, point, 0)

  private def insertRec(node: KDNode, point: KDPoint, depth: Int): KDNode = {
    if (node == null) newNode(point: KDPoint)
    else {
      val axis = depth % dist.globalDim
      val pointVal = dist.byGlobal(point.obj, axis)
      val nodeVal = dist.byGlobal(node.point.obj, axis)
      if (pointVal < nodeVal)
        node.left = insertRec(node.left, point, depth + 1)
      else if (matchPoints(node.point, point)) {
        node.point.cluster = point.cluster
        node.deleted = false
      }
      else
        node.right = insertRec(node.right, point, depth + 1)
      node
    }
  }

  private def matchPoints(point1: KDPoint, point2: KDPoint): Boolean = {
    Compares.equals(point1.obj, point2.obj)
  }

  def delete(point: KDPoint): KDNode =
    deleteRec(this.root, point, 0)

  private def deleteRec(node: KDNode, point: KDPoint, depth: Int): KDNode = {
    if (node == null)
      return node
    val axis = depth % dist.globalDim
    if (matchPoints(node.point, point))
      node.deleted = true
    else {
      if (dist.byGlobal(point.obj, axis) < dist.byGlobal(node.point.obj, axis))
        node.left = deleteRec(node.left, point, depth + 1)
      else
        node.right = deleteRec(node.right, point, depth + 1)
    }
    node
  }

  def closestPointOfOtherCluster(point: KDPoint): KDPoint = {
    val c = closestRec(this.root, point, 0)
    if (c == null) null
    else c.point
  }

  private def closestRec(node: KDNode, point: KDPoint, depth: Int): KDNode = {
    if (node == null)
      return null
    if (point.cluster == node.point.cluster)
      return closerDistance(point,
        closestRec(node.left, point, depth + 1),
        closestRec(node.right, point, depth + 1))

    val axis = depth % dist.globalDim
    if (dist.byGlobal(point.obj, axis) < dist.byGlobal(node.point.obj, axis)) {
      val best = {
        if (node.deleted) closestRec(node.left, point, depth + 1)
        else closerDistance(point, closestRec(node.left, point, depth + 1), node)
      }
      if (best == null || dist(point.obj, best.point.obj) > dist.approxDistByGlobal(point.obj, node.point.obj, axis))
        closerDistance(point, closestRec(node.right, point, depth + 1), best)
      else best
    } else {
      val best =
        if (node.deleted) closestRec(node.right, point, depth + 1)
        else closerDistance(point, closestRec(node.right, point, depth + 1), node)
      if (best == null || dist(point.obj, best.point.obj) > dist.approxDistByGlobal(point.obj, node.point.obj, axis))
        closerDistance(point, closestRec(node.left, point, depth + 1), best)
      else best
    }
  }

  private def closerDistance(pivot: KDPoint, n1: KDNode, n2: KDNode): KDNode = {
    if (n1 == null)
      return n2
    if (n2 == null)
      return n1

    val d1 = dist(pivot.obj, n1.point.obj)
    val d2 = dist(pivot.obj, n2.point.obj)
    if (d1 < d2) n1 else n2
  }
}

case class MinHeap(maxSize: Int) {
  private val data = new Array[CURECluster](maxSize)
  private var size = -1

  private def parent(index: Int): Int = index / 2

  private def leftChild(index: Int): Int = index * 2

  private def rightChild(index: Int): Int = index * 2 + 1

  def swap(a: Int, b: Int): Unit = {
    val tmp = data(a)
    data(a) = data(b)
    data(b) = tmp
  }

  def insert(cluster: CURECluster): Unit = {
    size += 1
    data(size) = cluster
    percolateUp(size)
  }

  def takeHead(): CURECluster = {
    val head = data(0)
    data(0) = data(size)
    data(size) = null
    size -= 1
    percolateDown(0)
    head
  }

  def update(index: Int, cluster: CURECluster): Unit = {
    data(index) = cluster
    heapify(index)
  }

  def remove(index: Int): Unit = {
    data(index) = data(size)
    size -= 1
    heapify(index)
  }

  def heapify(index: Int): Unit = {
    val parentI = parent(index)
    if (parentI > 0 && (data(parentI).distance > data(index).distance))
      percolateUp(index)
    else
      percolateDown(index)
  }

  def getDataArray: Array[CURECluster] =
    data

  def heapSize: Int =
    this.size + 1

  @tailrec
  private def percolateUp(curr: Int): Unit = {
    val parentI = parent(curr)
    if (data(parentI).distance > data(curr).distance) {
      swap(parentI, curr)
      percolateUp(parentI)
    }
  }

  @tailrec
  private def percolateDown(curr: Int): Unit = {
    val lChild = leftChild(curr)
    val rChild = rightChild(curr)

    var min = curr
    if (lChild <= size &&
      data(lChild).distance < data(curr).distance)
      min = lChild
    if (rChild <= size &&
      data(rChild).distance < data(min).distance)
      min = rChild

    if (min != curr) {
      swap(min, curr)
      percolateDown(min)
    }
  }
}