package decisiontreecsv


import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{RFormula, StringIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}

import java.time.Instant

//-----------------------------------Decision Tree with CSV--------------------------//

object Newcsvconnectionml {

  def main(args: Array[String]): Unit = {

    println("Apache application Started...")

    val spark = SparkSession.builder()
      .appName("Create DataFrame from CSV file")
      .master("local[*]")
//      .config("spark.sql.shuffle.partitions", "200")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val currentInstant: Instant = Instant.now() // Represents a point in time
    val currentEpochSeconds: Long = currentInstant.getEpochSecond()

    val source1 = "https://github.com/shankybit/newmachinelearningcsvandneo/blob/3524e590b1d8b47a63cd2660d74f8094a1966675/sales.csv"
    val users_df_1 = spark.read.options(Map("inferSchema" -> "true", "header" -> "true")).csv(source1)

    val source2 = "https://github.com/shankybit/newmachinelearningcsvandneo/blob/3524e590b1d8b47a63cd2660d74f8094a1966675/shoes.csv"
    val users_df_2 = spark.read.options(Map("inferSchema" -> "true", "header" -> "true")).csv(source2)

    val source3 = "https://github.com/shankybit/newmachinelearningcsvandneo/blob/3524e590b1d8b47a63cd2660d74f8094a1966675/inventory.csv"
    val users_df_3 = spark.read.options(Map("inferSchema" -> "true", "header" -> "true")).csv(source3)

    users_df_1.show(10, false)
    users_df_1.printSchema()

    users_df_2.show(10, false)
    users_df_2.printSchema()

    users_df_3.show(10, false)
    users_df_3.printSchema()


    val user_schema = StructType(Array(
      StructField("sale_id", IntegerType, true),
      StructField("branch_id", IntegerType, true),
      StructField("shoe_id", StringType, true),
      StructField("sale_date", StringType, true),
      StructField("quantity_sold", IntegerType, true),
      StructField("total_price", IntegerType, true),
    ))

    val shoe_schema = StructType(Array(
      StructField("shoe_id", StringType, true),
      StructField("category", StringType, true),
      StructField("color", StringType, true),
      StructField("size", IntegerType, true),
      StructField("price", IntegerType, true)
    ))

    val inventory_schema = StructType(Array(
      StructField("branch_id", IntegerType, true),
      StructField("shoe_id", StringType, true),
      StructField("quantity_in_stock", IntegerType, true),
      StructField("reorder_level", IntegerType, true),
    ))

    val users_pipe_read1 = spark.read.option("header", true).schema(user_schema).csv(source1)
    users_pipe_read1.show(10, false)
    users_pipe_read1.printSchema()

    val users_pipe_read2 = spark.read.option("header", true).schema(shoe_schema).csv(source2)
    users_pipe_read2.show(10, false)
    users_pipe_read2.printSchema()

    val users_pipe_read3 = spark.read.option("header", true).schema(inventory_schema).csv(source3)
    users_pipe_read3.show(10, false)
    users_pipe_read3.printSchema()
    val users_pipe_read3_renamed = users_pipe_read3.withColumnRenamed("branch_id", "b_id")

    var joinedDF = users_pipe_read1.join(users_pipe_read3_renamed, users_pipe_read1("shoe_id") === users_pipe_read3_renamed("shoe_id"), "inner")

    joinedDF.show()

    joinedDF = joinedDF.drop("shoe_id")
    joinedDF = joinedDF.drop("sale_id")
    //    joinedDF = joinedDF.drop("total_price")
    joinedDF = joinedDF.drop("reorder_level")
    joinedDF = joinedDF.drop("b_id")

    joinedDF = joinedDF.withColumn(
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

    //    joinedDF = joinedDF.drop("sale_date")
    joinedDF = joinedDF.dropDuplicates()
    joinedDF.show()


    //-------------------------------One-hot-encoding-------------------------------//


      val indexedDF = new StringIndexer().setInputCol("sale_date").setOutputCol("sale_date_index")

    var labelDF = indexedDF.fit(joinedDF).transform(joinedDF)

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


    println(secondDiff)


    spark.stop()

  }

}
