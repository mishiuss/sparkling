package ru.ifmo.rain.algorithms.birch

import ru.ifmo.rain.distances.MultiDistance

import scala.collection.mutable

class CFNode(val maxBranches: Int, val distThreshold: Double) extends Serializable {
  val entries: mutable.Set[CFEntry] = mutable.HashSet.empty[CFEntry]

  def addEntry(entry: CFEntry): Unit = entries += entry

  private def removeEntry(entry: CFEntry): Unit = entries -= entry

  private def closestEntryWith(entry: CFEntry)(implicit dist: MultiDistance): CFEntry =
    entries.minBy(e => dist(e.centroid, entry.centroid))

  def closestEntryPair(implicit dist: MultiDistance): Option[(CFEntry, CFEntry)] = {
    if (entries.size < 2) return None

    Option {
      {
        for {
          e1 <- entries
          e2 <- entries
          if e1.## < e2.##
        } yield (e1, e2)
      }.minBy { case (x, y) => x.distTo(y) }
    }
  }

  def farthestEntryPair(implicit dist: MultiDistance): Option[(CFEntry, CFEntry)] = {
    if (entries.size < 2) return None

    Option {
      {
        for {
          e1 <- entries
          e2 <- entries
          if e1.## < e2.##
        } yield (e1, e2)
      }.maxBy { case (x, y) => dist(x.centroid, y.centroid) }
    }
  }

  def split(implicit dist: MultiDistance): Option[(CFNode, CFNode)] =
    farthestEntryPair.map {
      case (farLeft, farRight) =>
        val leftNode = new CFNode(maxBranches, distThreshold)
        val rightNode = new CFNode(maxBranches, distThreshold)
        for (entry <- entries) {
          val left = dist(entry.centroid, farLeft.centroid)
          val right = dist(entry.centroid, farRight.centroid)
          if (left < right) leftNode.addEntry(entry)
          else rightNode.addEntry(entry)
        }
        (leftNode, rightNode)
    }

  def insertEntry(entry: CFEntry)(implicit dist: MultiDistance): Boolean = {
    if (entries.isEmpty) {
      addEntry(entry)
      return false
    }

    val closestEntry = closestEntryWith(entry)

    if (closestEntry.child != null) {
      val needSplit = closestEntry.child.insertEntry(entry)
      if (needSplit) {
        closestEntry.child.split.foreach {
          case (leftNode, rightNode) =>
            val newLeftEntry = new CFEntry(
              leftNode.entries.map(_.n).sum,
              leftNode.entries.map(_.ls).reduce(dist.plus(_, _)),
              leftNode
            )

            val newRightEntry = new CFEntry(
              rightNode.entries.map(_.n).sum,
              rightNode.entries.map(_.ls).reduce(dist.plus(_, _)),
              rightNode
            )

            removeEntry(closestEntry)
            addEntry(newLeftEntry)
            addEntry(newRightEntry)

            entries.size > maxBranches
        }
      }
      closestEntry.update(entry)
      false
    } else if (closestEntry.canMerge(entry, distThreshold)) {
      closestEntry.update(entry)
      false
    } else {
      addEntry(entry)
      entries.size > maxBranches
    }
  }
}

class CFLeafNode(maxBranches: Int, distThreshold: Double) extends CFNode(maxBranches, distThreshold) {
  var nextLeaf: CFLeafNode = null
  var prevLeaf: CFLeafNode = null

  override def split(implicit dist: MultiDistance): Option[(CFLeafNode, CFLeafNode)] =
    farthestEntryPair.map {
      case (farLeft, farRight) =>
        val leftNode = new CFLeafNode(maxBranches, distThreshold)
        val rightNode = new CFLeafNode(maxBranches, distThreshold)

        for (entry <- entries) {
          val left = dist(entry.centroid, farLeft.centroid)
          val right = dist(entry.centroid, farRight.centroid)
          if (left < right) leftNode.addEntry(entry)
          else rightNode.addEntry(entry)
        }

        leftNode.setNextLeaf(rightNode)
        if (prevLeaf != null) prevLeaf.setNextLeaf(leftNode)
        if (nextLeaf != null) rightNode.setNextLeaf(nextLeaf)

        (leftNode, rightNode)
    }

  def setNextLeaf(node: CFLeafNode): Unit = {
    nextLeaf = node
    if (node != null) {
      node.prevLeaf = this
    }
  }
}