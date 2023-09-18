package ru.ifmo.rain.algorithms.dbscan

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer


class SyncBuf [T] extends ArrayBuffer[T] with mutable.SynchronizedBuffer[T]
