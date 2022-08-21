package com.nazjara.streaming;

import org.apache.spark.SparkConf;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import scala.Tuple2;

public class LogStreamAnalysis {

    public static void main(String[] args) throws InterruptedException {
        var sparkConfig = new SparkConf()
                .setAppName("Spark streaming app")
                .setMaster("local[*]");

        try (var sparkContext = new JavaStreamingContext(sparkConfig, Durations.seconds(5))) {
            var inputStreamReceiver = sparkContext.socketTextStream("localhost", 8989);
            var inputStream =  inputStreamReceiver.map(item -> item);

            var pairInputStream = inputStream
                    .map(message -> message.split(",")[0])
                    .mapToPair(message -> new Tuple2<>(message, 1L))
                    .reduceByKeyAndWindow(Long::sum, Durations.minutes(60));

            pairInputStream.print();
            sparkContext.start();
            sparkContext.awaitTermination();
        }
    }
}
