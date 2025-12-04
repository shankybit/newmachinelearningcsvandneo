package decisiontreeneo4j


import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{RFormula, StringIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.functions.{col, when}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}
import org.neo4j.driver.{AuthTokens, Config, GraphDatabase, SessionConfig}

import java.time.Instant
import scala.collection.mutable.ListBuffer
import scala.jdk.CollectionConverters._

//-----------------------------------------Decision Tree with CSV-------------------//


object Newneo4jconnectionml {

  def main(args: Array[String]): Unit = {

    println("Apache application Started...")

    val spark = SparkSession.builder()
      .appName("Create DataFrame from Neo4j")
      .master("local[*]")
      .getOrCreate()


    //  val sc = spark.sparkContext
    //  val sql = spark.sqlContext

    case class Shoe(shoe_id:String, quantity_in_stock:Int, branch_id:Int, sale_date:String, quantity_sold:Int, total_price:Int)

    spark.sparkContext.setLogLevel("WARN")
    println("Will connect to Neo4j Database")


    val noSSL = Config.builder().build()
    val pwd = "neo4jdatabase"
    val user = "neo4j"
    val uri = "neo4j://127.0.0.1:7687"
    val db = "shoesalesdata"

    val driver = GraphDatabase.driver(uri, AuthTokens.basic(user, pwd), noSSL)

    val currentInstant: Instant = Instant.now() // Represents a point in time
    val currentEpochSeconds: Long = currentInstant.getEpochSecond()

    try {
      driver.verifyConnectivity()
      println("Connection established and verified.")
    } catch {
      case e: Exception =>
        println(s"Connection failed: ${e.getMessage}")
        driver.close()

    }
    var listBuffer = ListBuffer.empty[Shoe]

    // Run a query with the specified database name
    try {
      // Sessions handle the communication and are lightweight
      driver.session(SessionConfig.builder().withDatabase(db).build()).run {
        "MATCH (sh:Shoe_ID) MATCH path = (q:Quantity_IN_STOCK)<-[:QUANTITY_IN_STOCK_FOR_SHOE_ID]-(sh)-[:BRANCH_IDFROM_SHOE_ID]->(b:Branch_ID) MATCH path1 = (sd:Sale_DATE)<-[:SALE_DATE_FOR_SHOE_ID]-(sh)-[:QUANTITY_SOLD_FROM_SHOE_ID]->(qu:Quantity_SOLD) MATCH path2 = (t:Total_PRICE)<-[:TOTAL_PRICE_FROM_SHOE_ID]-(sh) RETURN sh, q, b, sd, qu, t;"

      }.asScala.foreach { record =>
        //      val shoe = record.get("shoe")
        //            println(s"Person name: ${movie.get("title")} - ${movie.get("released")} : ${movie.get("tagline")}")

        listBuffer.append(Shoe(s"${record.get("sh").asNode().values()}".replace("[","").replace("]","").toString, s"${record.get("q").asNode().values()}".replace("[","").replace("]","").toInt, s"${record.get("b").asNode().values()}".replace("[","").replace("]","").toInt, s"${record.get("sd").asNode().values()}".replace("[","").replace("]","").toString, s"${record.get("qu").asNode().values()}".replace("[","").replace("]","").toInt, s"${record.get("t").asNode().values()}".replace("[","").replace("]","").toInt))

      }

    } catch {
      case e: Exception => println(s"Query failed: ${e.getMessage}")
    } finally {
      // Close the driver when the application shuts down
      driver.close()

    }

    val schema = StructType(Array(
      StructField("shoe_id", StringType, true),
      StructField("quantity_in_stock", IntegerType, true),
      StructField("branch_id", IntegerType, true),
      StructField("sale_date", StringType, true),
      StructField("quantity_sold", IntegerType, true),
      StructField("total_price", IntegerType, true)
    )) // Required for toDF() on case class collections
    //  val newScalaList: List[Row] = javaList.asScala.toList

    val finalImmutableList = listBuffer.toList
//    val ImmutableList = listBuffer.toList.toDF()
    val rows = finalImmutableList.map(p => Row(p.shoe_id, p.quantity_in_stock, p.branch_id, p.sale_date, p.quantity_sold, p.total_price))
    //
    val rdd = spark.sparkContext.parallelize(rows)
    val rdDFWithSchema = spark.createDataFrame(rdd, schema)
    var newrddWithSchema = rdDFWithSchema.dropDuplicates()
    newrddWithSchema = newrddWithSchema.withColumn(
      "sale_date",
      when(col("sale_date").like("%-01-%"), "January")
        .when(col("sale_date").like("%-02-%"), "February")
        .when(col("sale_date").like("%-03-%"), "March")
        .when(col("sale_date").like("%-04-%"), "April")
        .when(col("sale_date").like("%-05-%"), "May")
        .when(col("sale_date").like("%-06-%"), "June")
        .when(col("sale_date").like("%-07-%"), "July")
        .when(col("sale_date").like("%-08-%"), "August")
        .when(col("sale_date").like("%-09-%"), "September")
        .when(col("sale_date").like("%-10-%"), "October")
        .when(col("sale_date").like("%-11-%"), "November")
        .when(col("sale_date").like("%-12-%"), "December")
    )

    newrddWithSchema = newrddWithSchema.drop("shoe_id")

    val indexedDF = new StringIndexer().setInputCol("sale_date").setOutputCol("sale_date_index")

    var labelDF = indexedDF.fit(newrddWithSchema).transform(newrddWithSchema)

    labelDF.show()


    val supervised = new RFormula()
      .setFormula("sale_date_index ~ . ")

    val fittedRF = supervised.fit(labelDF)
    val preparedDF = fittedRF.transform(labelDF)

    preparedDF.show(false)
    preparedDF.sample(0.1).show(false)

    //-----------------------------------------------------------------------------//

    import org.apache.spark.ml.feature.VectorAssembler

    //---------------------------------Vector Assembler----------------------------//

    val assembler = new VectorAssembler()
      .setInputCols(Array("features"))
      .setOutputCol("features_vec")

    val assembledData = assembler.transform(preparedDF)
    assembledData.show()


    // --------------------------------------Decision Tree-------------------------//

    val Array(train, test) = assembledData.randomSplit(Array(0.8, 0.2))

    import org.apache.spark.ml.classification.DecisionTreeClassifier


    val decTree = new DecisionTreeClassifier()
      .setLabelCol("sale_date_index")
      .setFeaturesCol("features_vec")

    //    val fittedModel = decTree.fit(train)
    //
    //    val testDF = fittedModel.transform(test)

    //----------------------------------------------dec tree--------------------------//

    //--------------------------------------Param-Builder---------------------------//

    // 3. Create a ParamGridBuilder
    val paramGrid = new ParamGridBuilder()
      .addGrid(decTree.maxDepth, Array(5, 15, 25)) // Example: try maxDepth of 5, 10, and 15
      .addGrid(decTree.maxBins, Array(2, 4, 8)) // Example: try maxBins of 16 and 32
      .build()

    //-------------------------------------Multi-classification---------------------//
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("sale_date_index")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    //-------------------------------------Cross-Validator--------------------------//

    // 4. Instantiate CrossValidator
    val cv = new CrossValidator()
      .setEstimator(decTree)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)


    val cvModel = cv.fit(train)
    val bestModel = cvModel.bestModel.asInstanceOf[org.apache.spark.ml.classification.DecisionTreeClassificationModel]

    // Print the best hyperparameters found
    println(s"Best maxDepth: ${bestModel.getMaxDepth}")
    println(s"Best maxBins: ${bestModel.getMaxBins}")

    val currentInstantOne: Instant = Instant.now() // Represents a point in time
    val currentEpochSecondsTwo: Long = currentInstantOne.getEpochSecond()

    val secondDiff: Long = currentEpochSecondsTwo - currentEpochSeconds

    val predictions = bestModel.transform(test)
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test set accuracy: $accuracy")

//    bestModel.write.overwrite.save("C:\\Users\\Admin\\workspace_scala\\newmachinelearningcsvandneo\\models\\mlmodelneodecTree")


    println(secondDiff)


    spark.close()

  }

}
