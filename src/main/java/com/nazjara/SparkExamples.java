package com.nazjara;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.sparkproject.guava.collect.Iterables;
import scala.Tuple2;
import scala.Tuple3;

import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.functions.*;

public class SparkExamples {

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

            System.out.println(dataset.first().getAs("subject").toString());
            System.out.println(Integer.parseInt(dataset.first().getAs("year")));

            //filter 1
            dataset
                    .filter("subject = 'Modern Art' and year >= 2007")
                    .show();

            //filter 2
            dataset
                    .filter((FilterFunction<Row>) row -> row.getAs("subject").toString().equals("Modern Art") &&
                            Integer.parseInt(row.getAs("year")) >= 2007)
                    .show();

            //filter 3
            dataset.filter(col("subject").equalTo("Modern Art")
                            .and(col("year").geq(2007)))
                    .show();

            //filter 4
            dataset.createOrReplaceTempView("students_view");
            sparkSession
                    .sql("select score, year from students_view where subject = 'Modern Art' and year >= 2007")
                    .show();

            //aggregations
            dataset
                    .groupBy(col("subject"))
                    .agg(max(col("score")).as("max_score"),
                         min(col("score")).as("min_score"))
                    .show();

            //pivot
            dataset
                    .groupBy(col("subject"))
                    .pivot(col("year"))
                    .agg(round(avg(col("score")), 2).as("average_score"),
                            round(stddev(col("score")), 2).as("standard_deviation_score"))
                    .show();

            //user-defined functions
            sparkSession.udf().register("hasPassed", (String grade, String subject) -> {

                if (subject.equals("Biology")){
                    return grade.startsWith("A");
                }

                return grade.startsWith("A") || grade.startsWith("B") || grade.startsWith("C");
            }, DataTypes.BooleanType);

            dataset
                    .withColumn("pass",
                            callUDF("hasPassed",
                                    col("grade"),
                                    col("subject")))
                    .show();

            //working with in-memory data
            var inMemoryData = List.of(
                    RowFactory.create("WARN", "2022-07-20 12:00:00"),
                    RowFactory.create("FATAL", "2022-07-19 13:00:00"),
                    RowFactory.create("WARN", "2022-07-18 14:00:00"),
                    RowFactory.create("INFO", "2022-08-17 15:00:00"),
                    RowFactory.create("WARN", "2022-08-16 16:00:00"));

            dataset = sparkSession.createDataFrame(inMemoryData, new StructType(new StructField[]{
                    new StructField("level", DataTypes.StringType, false, Metadata.empty()),
                    new StructField("datetime", DataTypes.StringType, false, Metadata.empty()),
            }));

            sqlExamples(dataset, sparkSession);
            dataFrameExamples(dataset);

            //working with files
            dataset = sparkSession.read()
                    .option("header", true)
                    .csv("src/main/resources/biglog.txt");

            sqlExamples(dataset, sparkSession);
            dataFrameExamples(dataset);
        }
    }

    private static void sqlExamples(Dataset<Row> dataset, SparkSession sparkSession)
    {
        dataset.createOrReplaceTempView("logging_view");
        dataset = sparkSession
                .sql("select level, " +
                        "collect_list(datetime) as datetime_list from logging_view " +
                        "group by level " +
                        "order by level");
        dataset.show(Integer.MAX_VALUE);

        sparkSession.udf().register("monthNum", (String month) -> {
            var inputDate = new SimpleDateFormat("MMMM").parse(month);
            return Integer.parseInt(new SimpleDateFormat("M").format(inputDate));
        }, DataTypes.IntegerType);

        dataset = sparkSession
                .sql("select level, " +
                        "date_format(datetime, 'MMMM') as month, " +
                        "count(1) as total from logging_view " +
                        "group by level, month " +
//                        "order by cast(first(date_format(datetime, 'M')) as int), level"); //cast instead of using UDF
                        "order by monthNum(month), level");
        dataset.show(Integer.MAX_VALUE);

        dataset.createOrReplaceTempView("final_view");
        sparkSession
                .sql("select sum(total) as total from final_view")
                .show(Integer.MAX_VALUE);
    }

    private static void dataFrameExamples(Dataset<Row> dataset)
    {
        var dataset2 = dataset
                .groupBy(col("level"))
                .agg(collect_list("datetime").as("datetime_list"))
                .orderBy(col("level"));
        dataset2.show(Integer.MAX_VALUE);

        var dataset3 = dataset
                .select(col("level"),
                        date_format(col("datetime"), "MMMM").as("month"),
                        date_format(col("datetime"), "M").as("monthnum").cast(DataTypes.IntegerType));

        var dataset4 = dataset3
                .groupBy(col("level"), col("month"), col("monthnum"))
                .count()
                .withColumnRenamed("count", "total")
                .orderBy(col("monthnum"), col("level"))
                .drop(col("monthnum"));
        dataset4.show(Integer.MAX_VALUE);

        dataset3
                .groupBy(col("level"))
                .pivot(col("month"), List.of("January", "February", "March", "April", "May" , "June",
                        "July", "August", "September", "October", "November", "December"))
                .count()
                .na()
                .fill(0)
                .show(Integer.MAX_VALUE);

        dataset4
                .select(sum(col("total")).as("total"))
                .show(Integer.MAX_VALUE);
    }
}
