package ru.ifmo.rain.algorithms.birch

import org.apache.spark.util.SizeEstimator
import ru.ifmo.rain.distances.MultiDistance

class CFTree(
              val maxBranches: Int,
              var distThreshold: Double,
              val dist: MultiDistance
            ) extends Iterable[CFEntry] with Serializable {

  private var autoBuild: Boolean = false
  private var memLimit: Long = (Runtime.getRuntime.totalMemory() * 0.61).toLong
  private var memCheckPeriod: Int = 8192
  private var tick: Int = 0

  private var root: CFNode = new CFLeafNode(maxBranches, distThreshold)
  private var leafDummyNode: CFLeafNode = new CFLeafNode(maxBranches, distThreshold)
  leafDummyNode.setNextLeaf(root.asInstanceOf[CFLeafNode])

  def enableAutoRebuild(): CFTree = {
    autoBuild = true
    this
  }

  def insertEntry(entry: CFEntry): Unit = {
    val needSplit = root.insertEntry(entry)(dist)
    if (needSplit) {
      root.split(dist).foreach {
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

          val newRoot = new CFNode(maxBranches, distThreshold)
          newRoot.addEntry(newLeftEntry)
          newRoot.addEntry(newRightEntry)

          root = newRoot
          System.gc()
      }
    }

    tick += 1
    if (autoBuild && tick == memCheckPeriod) {
      tick = 0
      rebuildIfAboveMemLimit()
    }
  }

  private def rebuildIfAboveMemLimit(): Unit = {
    val curMemUsage = SizeEstimator.estimate(root)
    if (curMemUsage <= memLimit) return
    val memRatio = curMemUsage / memLimit
    val newDistThreshold = computeGreaterThreshold(distThreshold * memRatio)
    rebuildTree(newDistThreshold)
  }

  private def computeGreaterThreshold(minThreshold: Double): Double = {
    var (estimatedT, count) = (0.0, 0)
    leafNodes.foreach { node =>
      node.closestEntryPair(dist).foreach { case (entry1, entry2) =>
        estimatedT += entry1.distTo(entry2)(dist)
        count += 1
      }
    }
    if (count > 0) estimatedT /= count
    val threshold = math.max(minThreshold, math.min(estimatedT, minThreshold * 2))

    if (threshold < distThreshold) distThreshold * 2
    else threshold
  }

  private def rebuildTree(newDistThreshold: Double): Unit = {
    val newTree = new CFTree(maxBranches, newDistThreshold, dist)
    newTree.autoBuild = autoBuild
    newTree.memLimit = memLimit
    newTree.memCheckPeriod = memCheckPeriod
    newTree.mergeTree(this)
    distThreshold = newDistThreshold
    root = newTree.root
    leafDummyNode = newTree.leafDummyNode
    System.gc()
  }

  private def mergeTree(that: CFTree): Unit = that.leafEntries.foreach(insertEntry)

  override def iterator: Iterator[CFEntry] = leafEntries

  private def leafEntries: Iterator[CFEntry] =
    for {
      node <- leafNodes
      e <- node.entries
    } yield e

  private def leafNodes: Iterator[CFNode] = new Iterator[CFNode] {
    var cur: CFLeafNode = leafDummyNode

    override def hasNext: Boolean = cur.nextLeaf != null

    override def next(): CFNode = {
      cur = cur.nextLeaf
      cur
    }
  }
}