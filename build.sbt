
ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.17"

lazy val root = (project in file("."))
  .settings(
    name := "newmachinelearningcsvandneo"
  )

// https://mvnrepository.com/artifact/org.apache.spark/spark-core
libraryDependencies += "org.apache.spark" %% "spark-core" % "3.4.3"

// https://mvnrepository.com/artifact/org.apache.spark/spark-sql
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.4.3"

// https://mvnrepository.com/artifact/org.neo4j.driver/neo4j-java-driver
libraryDependencies += "org.neo4j.driver" % "neo4j-java-driver" % "4.2.5"

// https://mvnrepository.com/artifact/org.apache.spark/spark-mllib
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.4.3"

// https://mvnrepository.com/artifact/com.dimafeng/neotypes
libraryDependencies += "com.dimafeng" %% "neotypes" % "0.17.0"

// https://mvnrepository.com/artifact/org.apache.spark/spark-graphx
libraryDependencies += "org.apache.spark" %% "spark-graphx" % "3.4.3"

// https://mvnrepository.com/artifact/org.neo4j/neo4j-connector-apache-spark
//libraryDependencies += "org.neo4j" %% "neo4j-connector-apache-spark" % "5.2.0_for_spark_3"

// https://mvnrepository.com/artifact/org.opencypher/morpheus-spark-cypher
//libraryDependencies += "org.opencypher" % "morpheus-spark-cypher" % "0.4.2"

// https://mvnrepository.com/artifact/org.neo4j/neo4j-connector-apache-spark
libraryDependencies += "org.neo4j" %% "neo4j-connector-apache-spark" % "5.1.0_for_spark_3"

// https://mvnrepository.com/artifact/org.neo4j/neo4j-graphdb-api
libraryDependencies += "org.neo4j" % "neo4j-graphdb-api" % "4.2.5"

libraryDependencies += "org.apache.spark" %% "spark-streaming" % "3.4.3"
libraryDependencies += "org.apache.spark" %% "spark-hive" % "3.4.3"