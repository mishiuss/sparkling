package ru.ifmo.rain.algorithms.dbscan

class Settings(
                val epsilon: Double,
                val minPoints: Long,
                val pointsInBox: Long,
                val axisSplits: Int,
                val levels: Int,
                val borderNoise: Boolean
              ) extends Serializable
