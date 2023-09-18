package ru.ifmo.rain.algorithms.dbscan.box

import ru.ifmo.rain.algorithms.dbscan.SyncBuf
import ru.ifmo.rain.algorithms.dbscan.point.Point

import scala.collection.mutable.ArrayBuffer


abstract class BoxTreeItemBase[T <: BoxTreeItemBase[_]](val box: MultiModalBox) extends Serializable {
  var children: List[T] = Nil

  def flatten[X <: BoxTreeItemBase[_]]: Iterable[X] = this.asInstanceOf[X] :: children.flatMap { _.flatten[X] }

  def flattenBoxes: Iterable[MultiModalBox] = flatten[BoxTreeItemBase[T]].map { _.box }

  def flattenBoxes(predicate: T => Boolean): Iterable [MultiModalBox] = {
    val result = ArrayBuffer[MultiModalBox]()
    flattenBoxes(predicate, result)
    result
  }

  private def flattenBoxes[X <: BoxTreeItemBase[_]](predicate: X => Boolean, buffer: ArrayBuffer[MultiModalBox]): Unit = {
    if (children.nonEmpty && children.exists { x => predicate(x.asInstanceOf[X]) }) {
      children.foreach { x => x.flattenBoxes[X](predicate, buffer) }
    }
    else buffer += this.box
  }
}

class BoxTreeItemWithCount(box: MultiModalBox) extends BoxTreeItemBase[BoxTreeItemWithCount](box) {
  var numberOfPoints: Long = 0

  override def clone(): BoxTreeItemWithCount  = {
    val result = new BoxTreeItemWithCount (this.box)
    result.children = this.children.map { x => x.clone() }
    result
  }

}

class BoxTreeItemWithPoints(
                             box: MultiModalBox,
                             val points: SyncBuf[Point] = new SyncBuf[Point](),
                             val adjacentBoxes: SyncBuf[BoxTreeItemWithPoints] = new SyncBuf[BoxTreeItemWithPoints]()
                           ) extends BoxTreeItemBase[BoxTreeItemWithPoints](box)


