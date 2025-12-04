package randomforestneo4j

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{RFormula, StringIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.functions.{col, when}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}
import org.neo4j.driver.{AuthTokens, Config, GraphDatabase, SessionConfig}

import java.time.Instant
import scala.collection.mutable.ListBuffer
import scala.jdk.CollectionConverters.asScalaIteratorConverter

//--------------------------------------Random Forest with Neo4j-----------------------//

object Newneo4jconnectionrandomforest {


  case class Shoesales(sale_id:Int, branch_id:Int, shoe_id:String, sale_date:String, quantity_sold:Int, total_price:Int, category:String, color:String, size:Int, price:Int)

  def main(args: Array[String]): Unit = {

    println("Apache application Started...")

    val spark = SparkSession.builder()
      .appName("Create DataFrame from Neo4j")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    //  val sc = spark.sparkContext
    //  val sql = spark.sqlContext


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
    var listBuffer = ListBuffer.empty[Shoesales]

    // Run a query with the specified database name
    try {
      // Sessions handle the communication and are lightweight
      driver.session(SessionConfig.builder().withDatabase(db).build()).run {
        "MATCH (sid:Sale_ID), (sh:Shoe_ID) MATCH path = (b:Branch_ID)<-[:BRANCH_ID_FROM_SALE_ID]-(sid)-[:SHOE_ID_MADE_SALE]->(sh:Shoe_ID) MATCH path1 = (sd:Sale_DATE)<-[:SALE_DATE_FROM_SALE_ID]-(sid)-[:QUANTITY_SOLDFROM_SALE_ID]->(q:Quantity_SOLD) MATCH path2 = (t:Total_PRICE)<-[:TOTAL_PRICE_FROM_SALE_ID]-(sid) MATCH path4 = (c:Category)<-[:FIND_CATEGORY]-(sh:Shoe_ID)-[:FIND_COLOR]->(clr:Color) MATCH path5 = (s:Size)<-[:FIND_SIZE]-(sh:Shoe_ID)-[:FIND_PRICE]->(p:Price) RETURN sid, b, sh, sd, q, t, c, clr, s, p;"

      }.asScala.foreach { record =>


        listBuffer.append(Shoesales(s"${record.get("sid").asNode().values()}".replace("[","").replace("]","").toInt, s"${record.get("b").asNode().values()}".replace("[","").replace("]","").toInt, s"${record.get("sh").asNode().values()}".replace("[","").replace("]","").toString, s"${record.get("sd").asNode().values()}".replace("[","").replace("]","").toString, s"${record.get("q").asNode().values()}".replace("[","").replace("]","").toInt, s"${record.get("t").asNode().values()}".replace("[","").replace("]","").toInt, s"${record.get("c").asNode().values()}".replace("[","").replace("]","").toString, s"${record.get("clr").asNode().values()}".replace("[","").replace("]","").toString, s"${record.get("s").asNode().values()}".replace("[","").replace("]","").toInt, s"${record.get("p").asNode().values()}".replace("[","").replace("]","").toInt))

      }

    } catch {
      case e: Exception => println(s"Query failed: ${e.getMessage}")
    } finally {
      // Close the driver when the application shuts down
      driver.close()

    }

    val schema = StructType(Array(
      StructField("sale_id", IntegerType, true),
      StructField("branch_id", IntegerType, true),
      StructField("shoe_id", StringType, true),
      StructField("sale_date", StringType, true),
      StructField("quantity_sold", IntegerType, true),
      StructField("total_price", IntegerType, true),
      StructField("category", StringType, true),
      StructField("color", StringType, true),
      StructField("size", IntegerType, true),
      StructField("price", IntegerType, true)
    ))


    val finalImmutableList = listBuffer.toList
    //    val ImmutableList = listBuffer.toList.toDF()
    val rows = finalImmutableList.map(p => Row(p.sale_id, p.branch_id, p.shoe_id, p.sale_date, p.quantity_sold, p.total_price, p.category, p.color, p.size, p.price))
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
    newrddWithSchema = newrddWithSchema.drop("sale_id")

    newrddWithSchema = newrddWithSchema.dropDuplicates()
    newrddWithSchema.show()


    //-------------------------------One-hot-encoding-------------------------------//


    val columnsToEncode = Array("sale_date", "category", "color")

    val indexers = columnsToEncode.map { colName =>
      new StringIndexer()
        .setInputCol(colName)
        .setOutputCol(s"${colName}_index")
    }

    val pipeline = new Pipeline().setStages(indexers)
    val model = pipeline.fit(newrddWithSchema)
    val indexedDF = model.transform(newrddWithSchema)
    indexedDF.show()


    val supervised = new RFormula()
      .setFormula("color_index ~ . ")

    val fittedRF = supervised.fit(indexedDF)
    val preparedDF = fittedRF.transform(indexedDF)

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

    //---------------------------------Random-Forest------------------------------//
    //
    //      // Create a RandomForestClassifier
    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features_vec")
      .setNumTrees(7) // Number of trees in the forest
    //


    // --------------------------------------tvs-------------------------//

    val Array(train, test) = assembledData.randomSplit(Array(0.7, 0.3))



    //--------------------------------------Param-Builder---------------------------//

    // 3. Create a ParamGridBuilder
    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.maxDepth, Array(10, 15, 25)) // Example: try maxDepth of 5, 10, and 15
      .addGrid(rf.maxBins, Array(32, 64, 128)) // Example: try maxBins of 16 and 32
      .build()

    //-------------------------------------Multi-classification---------------------//
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("color_index")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    //-------------------------------------Cross-Validator--------------------------//

    // 4. Instantiate CrossValidator
    val cv = new CrossValidator()
      .setEstimator(rf)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(7)


    val cvModel = cv.fit(train)
    val bestModel = cvModel.bestModel.asInstanceOf[org.apache.spark.ml.classification.RandomForestClassificationModel]

    // Print the best hyperparameters found
    println(s"Best maxDepth: ${bestModel.getMaxDepth}")
    println(s"Best maxBins: ${bestModel.getMaxBins}")

    val currentInstantOne: Instant = Instant.now() // Represents a point in time
    val currentEpochSecondsTwo: Long = currentInstantOne.getEpochSecond()

    val secondDiff: Long = currentEpochSecondsTwo - currentEpochSeconds

    val predictions = bestModel.transform(test)
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test set accuracy: $accuracy")


    println(secondDiff)


    spark.close()

  }

}
