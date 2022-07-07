package com.nazjara;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.sparkproject.guava.collect.Iterables;
import scala.Tuple2;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class Main {

    public static void main(String[] args) {
        var sparkConfig = new SparkConf()
                .setAppName("Spark app")
                .setMaster("local[*]");

        try (var sparkContext = new JavaSparkContext(sparkConfig)) {
            var rdd1 = sparkContext.parallelize(List.of(11L, 12L, 13L, 14L));
            System.out.println(rdd1.reduce(Long::sum)); //sum
            var sqrtRdd = rdd1.map(value -> new Tuple2<>(value, Math.sqrt(value))); //square root
            sqrtRdd.collect().forEach(tuple -> System.out.println(tuple._1() + ":" + tuple._2())); //print all elements
            System.out.println(sqrtRdd.count()); //count1
            System.out.println(sqrtRdd.map(i -> 1L).reduce(Long::sum)); //count2

            // reduceByKey
            sparkContext.parallelize(
                    List.of("WARN: log message 1",
                            "ERROR: log message 2",
                            "FATAL: log message 3",
                            "ERROR: log message 4",
                            "WARN: log message 5"))
                    .mapToPair(value -> new Tuple2<>(value.split(":")[0], 1L))
                    .reduceByKey(Long::sum)
                    .collect()
                    .forEach(tuple -> System.out.println(tuple._1() + ":" + tuple._2()));

            //groupByKey (performance issues)
            sparkContext.parallelize(
                            List.of("WARN: log message 1",
                                    "ERROR: log message 2",
                                    "FATAL: log message 3",
                                    "ERROR: log message 4",
                                    "WARN: log message 5"))
                    .mapToPair(value -> new Tuple2<>(value.split(":")[0], 1L))
                    .groupByKey()
                    .collect()
                    .forEach(tuple -> System.out.println(tuple._1() + ":" + Iterables.size(tuple._2())));

            //flatMap + filter
            sparkContext.parallelize(
                    List.of("WARN: log message 1",
                            "ERROR: log message 2",
                            "FATAL: log message 3",
                            "ERROR: log message 4",
                            "WARN: log message 5"))
                    .flatMap(value -> Arrays.asList(value.split(" ")).iterator())
                    .filter(value -> value.length() > 1)
                    .collect()
                    .forEach(System.out::println);

        }
    }
}
