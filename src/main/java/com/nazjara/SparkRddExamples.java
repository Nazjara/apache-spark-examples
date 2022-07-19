package com.nazjara;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.sparkproject.guava.collect.Iterables;
import scala.Tuple2;
import scala.Tuple3;

import java.util.Arrays;
import java.util.List;

public class SparkRddExamples {

    public static void main(String[] args) {
        sparkRddExamples();
        sparkSqlExamples();
    }

    private static void sparkRddExamples() {
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
                    .forEach(tuple -> System.out.println(tuple._1() + " : " + tuple._2()));

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
                    .forEach(tuple -> System.out.println(tuple._1() + " : " + Iterables.size(tuple._2())));

            // count most used not boring words
            sparkContext.textFile("src/main/resources/subtitles/input-spring.txt")
                    .flatMap(value -> Arrays.asList(value.split(" ")).iterator())
                    .mapToPair(value -> new Tuple2<>(value.toLowerCase().replaceAll("[^a-zA-Z]", ""), 1L))
                    .filter(value -> !value._1().isBlank() && Util.isNotBoring(value._1()))
                    .reduceByKey(Long::sum)
                    .mapToPair(Tuple2::swap)
                    .sortByKey(false)
//                    .cache()
//                    .persist(StorageLevel.MEMORY_AND_DISK())
                    .take(10)
                    .forEach(System.out::println);

            countCourseScores(sparkContext);
        }
    }

    private static void countCourseScores(JavaSparkContext sparkContext) {
        var userChapter = setUpViewDataRdd(sparkContext);
        var chapterCourse = setUpChapterDataRdd(sparkContext);
        var titlesData = setUpTitlesDataRdd(sparkContext);

        chapterCourse
                .fullOuterJoin(userChapter
                        .distinct()
                        .mapToPair(Tuple2::swap))
                .mapToPair(tuple ->
                        new Tuple2<>(tuple._2()._1().get(),
                                new Tuple2<>(tuple._1(),
                                        tuple._2()._2().orElse(0))))
                .sortByKey(true)
                .fullOuterJoin(chapterCourse
                        .mapToPair(tuple -> new Tuple2<>(tuple._2(), 1))
                        .reduceByKey(Integer::sum))
                .mapToPair(tuple ->
                        new Tuple2<>(
                                new Tuple3<>(tuple._1(), tuple._2()._1().get()._2(), tuple._2()._2().get()), 1))
                .reduceByKey(Integer::sum)
                .mapToPair(tuple -> {
                    var score = (double) tuple._2() / tuple._1()._3();

                    if (score > 0.9) {
                        score = 10;
                    } else if (score > 0.5) {
                        score = 4;
                    } else if (score > 0.25) {
                        score = 2;
                    } else {
                        score = 0;
                    }

                    return new Tuple2<>(
                            new Tuple2<>(tuple._1()._1(), tuple._1()._2()), (int) score);
                })
                .filter(tuple -> tuple._1()._2() != 0)
                .mapToPair(tuple -> new Tuple2<>(tuple._1()._1(), tuple._2()))
                .reduceByKey(Integer::sum)
                .join(titlesData)
                .mapToPair(tuple -> new Tuple2<>(tuple._2()._1(), tuple._2()._2()))
                .sortByKey(false)
                .collect()
                .forEach(System.out::println);
    }

    private static JavaPairRDD<Integer, String> setUpTitlesDataRdd(JavaSparkContext sc) {
        return sc.textFile("src/main/resources/viewing figures/titles.csv")
                .mapToPair(commaSeparatedLine -> {
                    var cols = commaSeparatedLine.split(",");
                    return new Tuple2<>(Integer.valueOf(cols[0]), cols[1]);
                });
    }

    private static JavaPairRDD<Integer, Integer> setUpChapterDataRdd(JavaSparkContext sc) {
        return sc.textFile("src/main/resources/viewing figures/chapters.csv")
                .mapToPair(commaSeparatedLine -> {
                    var cols = commaSeparatedLine.split(",");
                    return new Tuple2<>(Integer.valueOf(cols[0]), Integer.valueOf(cols[1]));
                });
    }

    private static JavaPairRDD<Integer, Integer> setUpViewDataRdd(JavaSparkContext sc) {
        return sc.textFile("src/main/resources/viewing figures/views-*.csv")
                .mapToPair(commaSeparatedLine -> {
                    var columns = commaSeparatedLine.split(",");
                    return new Tuple2<>(Integer.valueOf(columns[0]), Integer.valueOf(columns[1]));
                });
    }

    private static void sparkSqlExamples() {
        try (var sparkSession = SparkSession.builder()
                .appName("Spark app")
                .master("local[*]")
                .getOrCreate()) {

            var dataset = sparkSession.read()
                    .option("header", true)
                    .csv("src/main/resources/exams/students.csv");

            dataset.show();
            System.out.println(dataset.count());
        }
    }
}
