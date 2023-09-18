ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.11.12"

lazy val root = (project in file("."))
  .settings(
    name := "heaven"
  )

artifactName := { (sv: ScalaVersion, module: ModuleID, artifact: Artifact) =>
  artifact.name + "." + artifact.extension
}

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % "2.4.6" % "provided",
  "org.apache.spark" %% "spark-mllib" % "2.4.6" % "provided"
)

libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.13" % "test"
