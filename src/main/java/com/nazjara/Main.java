package com.nazjara;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.List;

public class Main {

    public static void main(String[] args) {
        var sparkConfig = new SparkConf()
                .setAppName("Spark app")
                .setMaster("local[*]");

        try (var sparkContext = new JavaSparkContext(sparkConfig))
        {
            var rdd = sparkContext.parallelize(List.of(11L, 12L, 13L, 14L));
            System.out.println(rdd.reduce(Long::sum)); //sum
            var sqrtRdd = rdd.map(value -> new Tuple2<>(value, Math.sqrt(value))); //square root
            sqrtRdd.collect().forEach(tuple -> System.out.println(tuple._1() + ":" + tuple._2())); //print all elements
            System.out.println(sqrtRdd.count()); //count1
            System.out.println(sqrtRdd.map(i -> 1L).reduce(Long::sum)); //count2
        }
    }
}
