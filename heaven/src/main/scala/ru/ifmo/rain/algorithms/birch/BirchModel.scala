package ru.ifmo.rain.algorithms.birch

import org.apache.spark.sql.Row
import ru.ifmo.rain.Sparkling
import ru.ifmo.rain.algorithms.kmeans.KMeansModel
import ru.ifmo.rain.distances.MultiDistance


@Sparkling
class BirchModel(centroids: Array[Row], dist: MultiDistance) extends KMeansModel(centroids, dist)
