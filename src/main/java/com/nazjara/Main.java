package com.nazjara;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import java.util.List;

public class Main {

    public static void main(String[] args) {
        var sparkConfig = new SparkConf()
                .setAppName("Spark app")
                .setMaster("local[*]");

        try (var sparkContext = new JavaSparkContext(sparkConfig))
        {
            var rdd = sparkContext.parallelize(List.of(11.1, 12.1, 13.1, 14.1));
            System.out.println(rdd.reduce(Double::sum));
        }
    }
}
