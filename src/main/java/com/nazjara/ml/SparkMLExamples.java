package com.nazjara.ml;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

import java.util.Arrays;

import static org.apache.spark.sql.functions.*;

public class SparkMLExamples {

    public static void main(String[] args) {
        try (var sparkSession = SparkSession.builder()
                .appName("Spark ML app")
                .master("local[*]")
                .getOrCreate()) {

            sparkSession.sparkContext().setLogLevel("WARN");

            gymRepsAnalysis(sparkSession);
            housePriceAnalysis(sparkSession);
            subscriptionManagement(sparkSession);
            freeTrialsManagement(sparkSession);
            gymRepsAnalysis2(sparkSession);
            courseRecommendations(sparkSession);
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
                // 1 - customers watched no videos, 0 - customers watched some videos
                .withColumn("next_month_views", when(col("next_month_views")
                        .$greater(0), 0).otherwise(1))
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

        var logisticRegression = new LogisticRegression();
        var paramGridBuilder = new ParamGridBuilder();
        var paramMap = paramGridBuilder
                .addGrid(logisticRegression.regParam(), new double[] {0.01, 0.1, 0.3, 0.5, 0.7, 1})
                .addGrid(logisticRegression.elasticNetParam(), new double[] {0, 0.5, 1})
                .build();

        var trainValidationSplit = new TrainValidationSplit()
                .setEstimator(logisticRegression)
                .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
                .setEstimatorParamMaps(paramMap)
                .setTrainRatio(0.9);

        var pipeline = new Pipeline();
        pipeline.setStages(new PipelineStage[] {indexer, encoder, vectorAssembler, trainValidationSplit});
        var pipelineModel = pipeline.fit(trainingAndTestData);
        var model = (TrainValidationSplitModel) pipelineModel.stages()[3];
        var logisticRegressionModel = (LogisticRegressionModel) model.bestModel();

        var holdOutResults = pipelineModel.transform(holdOutData);
        holdOutResults.groupBy("label", "prediction").count().show();

        holdOutResults = holdOutResults.drop("prediction")
                .drop("rawPrediction")
                .drop("probability");

        System.out.println("The training data accuracy score (closer to 1 is better) is " +
                logisticRegressionModel.summary().accuracy());
        System.out.println(logisticRegressionModel.coefficients() + " " + logisticRegressionModel.intercept());
        System.out.println(logisticRegressionModel.getRegParam() + " " + logisticRegressionModel.getElasticNetParam());

        var summary = logisticRegressionModel.evaluate(holdOutResults);

        System.out.println("The testing data accuracy score (closer to 1 is better) is " + summary.accuracy());

        var truePositives = summary.truePositiveRateByLabel()[1];
        var falsePositives = summary.falsePositiveRateByLabel()[0];

        System.out.println("For the holdout data, the likelihood of the positive to be correct is " +
                (truePositives / (truePositives + falsePositives)));
    }

    private static Column getNullSubstitutionFunction(String columnName) {
        return when(col(columnName).isNull(), 0).otherwise(col(columnName));
    }

    private static void freeTrialsManagement(SparkSession sparkSession) {
        sparkSession.udf().register("countryGrouping", countryGrouping, DataTypes.StringType);

        var dataset = sparkSession.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/ml/freeTrials.csv");

        dataset = dataset
                .withColumn("country", callUDF("countryGrouping", col("country")))
                .withColumn("label", when(col("payments_made").geq(1), lit(1)).otherwise(lit(0)));

        dataset = new StringIndexer()
                .setInputCols(new String[] {"country"})
                .setOutputCols(new String[] {"countryIndex"})
                .fit(dataset)
                .transform(dataset);

        new IndexToString()
                .setInputCol("countryIndex")
                .setOutputCol("value")
                .transform(dataset.select("countryIndex").distinct())
                .show();

        dataset = new VectorAssembler()
                .setInputCols(new String[]{"countryIndex", "rebill_period", "chapter_access_count", "seconds_watched"})
                .setOutputCol("features")
                .transform(dataset)
                .select("label", "features");

        var combinedData = dataset.randomSplit(new double[] {0.8, 0.2});
        var trainingAndTestData = combinedData[0];
        var holdOutData = combinedData[1];

        var decisionTreeClassifier = new DecisionTreeClassifier();
        decisionTreeClassifier.setMaxDepth(3);

        var model = new DecisionTreeClassifier()
                .setMaxDepth(3)
                .fit(trainingAndTestData);

        dataset = model.transform(holdOutData);
        dataset.show();

        System.out.println(model.toDebugString());

        System.out.println("Model 1 accuracy is " + new MulticlassClassificationEvaluator()
                .setMetricName("accuracy")
                .evaluate(dataset));

        var randomForestClassifier = new RandomForestClassifier();
        randomForestClassifier.setMaxDepth(3);

        var model2 = new RandomForestClassifier()
                .setMaxDepth(3)
                .fit(trainingAndTestData);

        dataset = model2.transform(holdOutData);
        dataset.show();

        System.out.println(model2.toDebugString());

        System.out.println("Model 2 accuracy is " + new MulticlassClassificationEvaluator()
                .setMetricName("accuracy")
                .evaluate(dataset));
    }

    private static final UDF1<String,String> countryGrouping = country -> {
        var topCountries =  Arrays.asList("GB", "US", "UA", "UNKNOWN");
        var europeanCountries =  Arrays.asList("BE", "BG", "CZ", "DK", "DE", "EE", "IE", "EL", "ES", "FR",
                "HR", "IT", "CY", "LV", "LT", "LU", "HU", "MT", "NL", "AT", "PL", "PT", "RO", "SI", "SK", "FI", "SE",
                "CH", "IS", "NO", "LI", "EU");

        if (topCountries.contains(country)) {
            return country;
        }

        if (europeanCountries.contains(country)) {
            return "EUROPE";
        }

        else {
            return "OTHER";
        }
    };

    private static void gymRepsAnalysis2(SparkSession sparkSession) {
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
        dataset = vectorAssembler.setInputCols(new String[] {"GenderVector", "Age", "Height", "Weight", "NoOfReps"})
                .setOutputCol("features")
                .transform(dataset)
                .select("features");
        dataset.show();

        for (int noOfClusters = 2; noOfClusters < 9; noOfClusters++) {
            var model = new KMeans()
                    .setK(noOfClusters)
                    .fit(dataset);
            var dataset2 = model.transform(dataset);
            dataset2.show();

            Arrays.stream(model.clusterCenters())
                    .forEach(System.out::println);

            dataset2.groupBy("prediction").count().show();

            System.out.println("Number of clusters is "+ noOfClusters);
            System.out.println("SSE value is " + model.summary().trainingCost());
            System.out.println("Silhouette with squared euclidean distance is " + new ClusteringEvaluator().evaluate(dataset2));
        }
    }

    private static void courseRecommendations(SparkSession sparkSession) {
        var dataset = sparkSession.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/ml/courseViews.csv");

        dataset.printSchema();

        dataset = dataset
                .withColumn("proportionWatched", col("proportionWatched")
                .multiply(100));
//                .groupBy("userId")
//                .pivot("courseId")
//                .sum("proportionWatched");

        var model = new ALS()
                .setMaxIter(10)
                .setRegParam(0.1)
                .setUserCol("userId")
                .setItemCol("courseId")
                .setRatingCol("proportionWatched")
                .fit(dataset)
                .setColdStartStrategy("drop");

        var finalDataset = dataset;
        model.recommendForAllUsers(5).takeAsList(5).forEach(row -> {
            System.out.println("User id: " + row.getAs(0) + "; recommendation: " + row.getAs(1).toString());
            System.out.println("This user has already watched: ");
            finalDataset.filter("userId = " + row.getAs(0)).show();
        });
    }
}
