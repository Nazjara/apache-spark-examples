package com.nazjara.streaming;

import org.apache.kafka.common.serialization.StringDeserializer;
import org.apache.spark.SparkConf;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import org.apache.spark.streaming.kafka010.ConsumerStrategies;
import org.apache.spark.streaming.kafka010.KafkaUtils;
import org.apache.spark.streaming.kafka010.LocationStrategies;
import scala.Tuple2;

import java.util.List;
import java.util.Map;

public class ViewingFiguresDStreamVersion {

    public static void main(String[] args) throws InterruptedException {
        var sparkConfig = new SparkConf()
                .setAppName("Spark streaming app using DStream")
                .setMaster("local[*]");

        try (var sparkContext = new JavaStreamingContext(sparkConfig, Durations.seconds(1))) {
            var topics = List.of("viewrecords");
            var params = Map.<String, Object>of(
                    "bootstrap.servers", "localhost:9092",
                    "key.deserializer", StringDeserializer.class,
                    "value.deserializer", StringDeserializer.class,
                    "group.id", "someGroup",
                    "auto.offset.reset", "latest",
                    "enable.auto.commit", false);

            var stream = KafkaUtils.<String, String>createDirectStream(sparkContext,
                    LocationStrategies.PreferConsistent(),
                    ConsumerStrategies.Subscribe(topics, params));

            var dStream = stream
                    .mapToPair(item -> new Tuple2<>(item.value(), 5L))
                    .reduceByKeyAndWindow(Long::sum, Durations.minutes(60), Durations.minutes(1))
                    .mapToPair(Tuple2::swap)
                    .transformToPair(rdd -> rdd.sortByKey(false));

            dStream.print(50);

            sparkContext.start();
            sparkContext.awaitTermination();
        }
    }
}
