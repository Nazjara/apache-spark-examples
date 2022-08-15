package com.nazjara;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.when;

public class SparkMLExamples {

    public static void main(String[] args) {
        try (var sparkSession = SparkSession.builder()
                .appName("Spark ML app")
                .master("local[*]")
                .getOrCreate()) {

            sparkSession.sparkContext().setLogLevel("WARN");

//            gymRepsAnalysis(sparkSession);
//            housePriceAnalysis(sparkSession);
            subscriptionManagement(sparkSession);
        }
    }

    private static void gymRepsAnalysis(SparkSession sparkSession) {
        var dataset = sparkSession.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/ml/GymCompetition.csv");

        dataset.printSchema();
        dataset.show();

        var genderIndexer = new StringIndexer();
        genderIndexer.setInputCol("Gender");
        genderIndexer.setOutputCol("GenderIndex");
        dataset = genderIndexer.fit(dataset).transform(dataset);
        dataset.show();

        var genderEncoder = new OneHotEncoder();
        genderEncoder.setInputCol("GenderIndex");
        genderEncoder.setOutputCol("GenderVector");
        dataset = genderEncoder.fit(dataset).transform(dataset);
        dataset.show();

        var vectorAssembler = new VectorAssembler();
        vectorAssembler
                .setInputCols(new String[]{"Age", "Height", "Weight", "GenderVector"})
                .setOutputCol("features");

        var transformedDataset = vectorAssembler.transform(dataset)
                .select("NoOfReps", "features")
                .withColumnRenamed("NoOfReps", "label");

        transformedDataset.show();

        var linearRegression = new LinearRegression();
        var linearRegressionModel = linearRegression.fit(transformedDataset);
        System.out.println("Intercept: " + linearRegressionModel.intercept() + ", coefficients: " + linearRegressionModel.coefficients());

        linearRegressionModel.transform(transformedDataset).show();
    }

    private static void housePriceAnalysis(SparkSession sparkSession) {
        var dataset = sparkSession.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/ml/kc_house_data.csv");

        dataset.printSchema();
        dataset.show();

        calculateFeatureCorrelation(dataset);

        dataset = dataset.withColumn("sqft_above_percentage",
                col("sqft_above").divide(col("sqft_living")))
                .withColumnRenamed("price", "label");

        var combinedData = dataset.randomSplit(new double[] {0.8, 0.2});
        var trainingAndTestData = combinedData[0];
        var holdOutData = combinedData[1];

        var indexer = new StringIndexer();
        indexer.setInputCols(new String[] {"condition", "grade", "zipcode"});
        indexer.setOutputCols(new String[] {"conditionIndex", "gradeIndex", "zipcodeIndex"});

        var encoder = new OneHotEncoder();
        encoder.setInputCols(new String[] {"conditionIndex", "gradeIndex", "zipcodeIndex"});
        encoder.setOutputCols(new String[] {"conditionVector", "gradeVector", "zipcodeVector"});

        var vectorAssembler = new VectorAssembler();
        vectorAssembler
                .setInputCols(new String[]{"bedrooms", "bathrooms", "sqft_living", "sqft_above_percentage", "floors",
                        "conditionVector", "gradeVector", "zipcodeVector", "waterfront"})
                .setOutputCol("features");

        var linearRegression = new LinearRegression();
        var paramGridBuilder = new ParamGridBuilder();
        var paramMap = paramGridBuilder
                .addGrid(linearRegression.regParam(), new double[] {0.01, 0.1, 0.5})
                .addGrid(linearRegression.elasticNetParam(), new double[] {0, 0.5, 1})
                .build();

        var trainValidationSplit = new TrainValidationSplit()
                .setEstimator(linearRegression)
                .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
                .setEstimatorParamMaps(paramMap)
                .setTrainRatio(0.8);

        var pipeline = new Pipeline();
        pipeline.setStages(new PipelineStage[] {indexer, encoder, vectorAssembler, trainValidationSplit});
        var pipelineModel = pipeline.fit(trainingAndTestData);
        var model = (TrainValidationSplitModel) pipelineModel.stages()[3];
        var linearRegressionModel = (LinearRegressionModel) model.bestModel();

        var holdOutResults = pipelineModel.transform(holdOutData);
        holdOutResults.show();
        holdOutResults = holdOutResults.drop("prediction");

        System.out.println("The training data r2 value (closer to 1 is better) is " + linearRegressionModel.summary().r2() +
                " and the RMSE (smaller is better) is " + linearRegressionModel.summary().rootMeanSquaredError());

        System.out.println("The testing data r2 value (closer to 1 is better) is " + linearRegressionModel.evaluate(holdOutResults).r2() +
                " and the RMSE (smaller is better) is " + linearRegressionModel.evaluate(holdOutResults).rootMeanSquaredError());

        System.out.println(linearRegressionModel.coefficients() + " " + linearRegressionModel.intercept());
        System.out.println(linearRegressionModel.getRegParam() + " " + linearRegressionModel.getElasticNetParam());
    }

    private static void calculateFeatureCorrelation(Dataset<Row> dataset)
    {
        var testDataset = dataset.drop("id", "date", "waterfront", "view", "condition", "grade",
                "yr_renovated", "zipcode", "lat", "long", "sqft_lot", "sqft_lot15", "yr_built", "sqft_living15",
                "conditionIndex", "gradeIndex", "zipcodeIndex", "conditionVector", "gradeVector", "zipcodeVector");

        Arrays.stream(testDataset.columns()).forEach(column ->
                Arrays.stream(testDataset.columns()).forEach(column2 ->
                        System.out.println("The correlation between " + column + " and " + column2 + " is " +
                                dataset.stat().corr(column, column2))));
    }

    private static void subscriptionManagement(SparkSession sparkSession) {
        var dataset = sparkSession.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/ml/part-r-00000-d55d9fed-7427-4d23-aa42-495275510f78.csv",
                        "src/main/resources/ml/part-r-00001-d55d9fed-7427-4d23-aa42-495275510f78.csv",
                        "src/main/resources/ml/part-r-00002-d55d9fed-7427-4d23-aa42-495275510f78.csv",
                        "src/main/resources/ml/part-r-00003-d55d9fed-7427-4d23-aa42-495275510f78.csv");

        dataset = dataset.filter("is_cancelled = false");
        dataset = dataset.withColumn("firstSub", getNullSubstitutionFunction("firstSub"))
                .withColumn("all_time_views", getNullSubstitutionFunction("all_time_views"))
                .withColumn("last_month_views", getNullSubstitutionFunction("last_month_views"))
                .withColumn("next_month_views", getNullSubstitutionFunction("next_month_views"))
                .withColumnRenamed("next_month_views", "label");

        var combinedData = dataset.randomSplit(new double[] {0.9, 0.1});
        var trainingAndTestData = combinedData[0];
        var holdOutData = combinedData[1];

        var indexer = new StringIndexer();
        indexer.setInputCols(new String[] {"payment_method_type", "country", "rebill_period_in_months"});
        indexer.setOutputCols(new String[] {"payment_method_typeIndex", "countryIndex", "rebill_period_in_monthsIndex"});

        var encoder = new OneHotEncoder();
        encoder.setInputCols(new String[] {"payment_method_typeIndex", "countryIndex", "rebill_period_in_monthsIndex"});
        encoder.setOutputCols(new String[] {"payment_method_typeVector", "countryVector", "rebill_period_in_monthsVector"});

        var vectorAssembler = new VectorAssembler();
        vectorAssembler
                .setInputCols(new String[]{"payment_method_typeVector", "countryVector", "rebill_period_in_monthsVector",
                        "firstSub", "age", "all_time_views", "last_month_views"})
                .setOutputCol("features");

        var linearRegression = new LinearRegression();
        var paramGridBuilder = new ParamGridBuilder();
        var paramMap = paramGridBuilder
                .addGrid(linearRegression.regParam(), new double[] {0.01, 0.1, 0.3, 0.5, 0.7, 1})
                .addGrid(linearRegression.elasticNetParam(), new double[] {0, 0.5, 1})
                .build();

        var trainValidationSplit = new TrainValidationSplit()
                .setEstimator(linearRegression)
                .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
                .setEstimatorParamMaps(paramMap)
                .setTrainRatio(0.9);

        var pipeline = new Pipeline();
        pipeline.setStages(new PipelineStage[] {indexer, encoder, vectorAssembler, trainValidationSplit});
        var pipelineModel = pipeline.fit(trainingAndTestData);
        var model = (TrainValidationSplitModel) pipelineModel.stages()[3];
        var linearRegressionModel = (LinearRegressionModel) model.bestModel();

        var holdOutResults = pipelineModel.transform(holdOutData);
        holdOutResults.show();
        holdOutResults = holdOutResults.drop("prediction");

        System.out.println("The training data r2 value (closer to 1 is better) is " + linearRegressionModel.summary().r2() +
                " and the RMSE (smaller is better) is " + linearRegressionModel.summary().rootMeanSquaredError());

        System.out.println("The testing data r2 value (closer to 1 is better) is " + linearRegressionModel.evaluate(holdOutResults).r2() +
                " and the RMSE (smaller is better) is " + linearRegressionModel.evaluate(holdOutResults).rootMeanSquaredError());

        System.out.println(linearRegressionModel.coefficients() + " " + linearRegressionModel.intercept());
        System.out.println(linearRegressionModel.getRegParam() + " " + linearRegressionModel.getElasticNetParam());
    }

    private static Column getNullSubstitutionFunction(String columnName) {
        return when(col(columnName).isNull(), 0).otherwise(col(columnName));
    }

}
