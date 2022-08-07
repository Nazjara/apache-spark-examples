package com.nazjara;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
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
        var trainingData = combinedData[0];
        var testData = combinedData[1];

        var model = new LinearRegression().fit(trainingData);

        System.out.println("The training data r2 value (closer to 1 is better) is " + model.summary().r2() +
                " and the RMSE (smaller is better) is " + model.summary().rootMeanSquaredError());

        model.transform(testData).show();

        System.out.println("The testing data r2 value (closer to 1 is better) is " + model.evaluate(testData).r2() +
                " and the RMSE (smaller is better) is " + model.evaluate(testData).rootMeanSquaredError());
    }
}
