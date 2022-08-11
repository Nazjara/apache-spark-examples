package com.nazjara;

import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.sql.SparkSession;

public class SparkMLExamples {

    public static void main(String[] args) {
        try (var sparkSession = SparkSession.builder()
                .appName("Spark ML app")
                .master("local[*]")
                .getOrCreate()) {

            gymRepsAnalysis(sparkSession);
            housePriceAnalysis(sparkSession);
        }

    }

    private static void gymRepsAnalysis(SparkSession sparkSession) {
        var dataset = sparkSession.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/ml/GymCompetition.csv");

        dataset.printSchema();
        dataset.show();

        var vectorAssembler = new VectorAssembler();
        vectorAssembler
                .setInputCols(new String[]{"Age", "Height", "Weight"})
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

        var vectorAssembler = new VectorAssembler();
        vectorAssembler
                .setInputCols(new String[]{"bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "grade"})
                .setOutputCol("features");

        var transformedDataset = vectorAssembler.transform(dataset)
                .select("price", "features")
                .withColumnRenamed("price", "label");

        transformedDataset.show();

        var combinedData = transformedDataset.randomSplit(new double[] {0.8, 0.2});
        var trainingAndTestData = combinedData[0];
        var holdOutData = combinedData[1];

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

        var model = (LinearRegressionModel) trainValidationSplit
                .fit(trainingAndTestData)
                .bestModel();

        System.out.println("The training data r2 value (closer to 1 is better) is " + model.summary().r2() +
                " and the RMSE (smaller is better) is " + model.summary().rootMeanSquaredError());

        model.transform(holdOutData).show();

        System.out.println("The testing data r2 value (closer to 1 is better) is " + model.evaluate(holdOutData).r2() +
                " and the RMSE (smaller is better) is " + model.evaluate(holdOutData).rootMeanSquaredError());

        System.out.println(model.coefficients() + " " + model.intercept());
        System.out.println(model.getRegParam() + " " + model.getElasticNetParam());
    }
}
