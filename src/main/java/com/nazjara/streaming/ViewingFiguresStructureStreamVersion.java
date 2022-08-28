package com.nazjara.streaming;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.streaming.OutputMode;
import org.apache.spark.sql.streaming.StreamingQueryException;
import org.apache.spark.sql.types.DataTypes;

import java.util.Map;
import java.util.concurrent.TimeoutException;

import static org.apache.spark.sql.functions.*;

public class ViewingFiguresStructureStreamVersion {

    public static void main(String[] args) throws InterruptedException, TimeoutException, StreamingQueryException {
        try (var sparkSession = SparkSession.builder()
                .appName("Spark streaming app using Structured Stream")
                .master("local[*]")
                .getOrCreate()) {

            sparkSession.conf().set("spark.sql.shuffle.partitions", "10");
            sparkSession.sparkContext().setLogLevel("WARN");

            var dataframe = sparkSession.readStream()
                    .format("kafka")
                    .option("kafka.bootstrap.servers", "localhost:9092")
                    .option("subscribe", "viewrecords")
                    .load();

//            var results = sparkStructuredStreamingSqlExample(sparkSession, dataframe);
            var results = sparkStructuredStreamingDataFrameExample(dataframe);

            var query = results.writeStream()
                    .format("console")
                    .outputMode(OutputMode.Update())
                    .option("truncate", false)
                    .option("numRows", 50)
                    .start();

            query.awaitTermination();
        }
    }

    private static Dataset<Row> sparkStructuredStreamingSqlExample(SparkSession sparkSession, Dataset<Row> dataframe) {
        dataframe.createOrReplaceTempView("viewing_figures");
        return sparkSession
                .sql("select window, cast(value as string) as course_name, sum(5) as seconds_watched from viewing_figures" +
                        " group by window(timestamp, '2 minutes'), course_name");
    }

    private static Dataset<Row> sparkStructuredStreamingDataFrameExample(Dataset<Row> dataframe) {
        return dataframe
                .withWatermark("timestamp", "10 minutes")
                .select(
                        window(col("timestamp").as("window"), "2 minutes"),
                        col("value").cast(DataTypes.StringType).as("course_name"))
                .groupBy(col("window"), col("course_name"))
                .agg(count("course_name").multiply(5).as("seconds_watched"));
    }
}
