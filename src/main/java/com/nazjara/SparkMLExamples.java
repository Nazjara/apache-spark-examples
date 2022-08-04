package com.nazjara;

import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.feature.VectorAssembler;

public class SparkMLExamples {

    public static void main(String[] args) {
        try (var sparkSession = SparkSession.builder()
                .appName("Spark ML app")
                .master("local[*]")
                .getOrCreate()) {

            var dataset = sparkSession.read()
                    .option("header", true)
                    .option("inferSchema", true)
                    .csv("src/main/resources/ml/GymCompetition.csv");

            dataset.printSchema();
            dataset.show();

            var vectorAssembler = new VectorAssembler();
            vectorAssembler.setInputCols(new String[]{"Age", "Height", "Weight"});
            vectorAssembler.setOutputCol("features");

            var transformedDataset = vectorAssembler.transform(dataset)
                    .select("NoOfReps", "features")
                    .withColumnRenamed("NoOfReps", "label");

            transformedDataset.show();

            var linearRegression = new LinearRegression();
            var linearRegressionModel = linearRegression.fit(transformedDataset);
            System.out.println("Intercept: " + linearRegressionModel.intercept() + ", coefficients: " + linearRegressionModel.coefficients());

            linearRegressionModel.transform(transformedDataset).show();
        }

    }
}
