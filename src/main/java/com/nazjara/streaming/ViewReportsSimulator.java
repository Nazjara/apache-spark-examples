package com.nazjara.streaming;

import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerRecord;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Map;
import java.util.Properties;
import java.util.Scanner;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class ViewReportsSimulator {

    private static final Map<Integer, String> COURSE_KEYS = Stream.of(new Object[][] {
                    {0, "Spring Boot Microservices"},
                    {1, "Spring Framework Fundamentals"},
                    {2, "Spring JavaConfig"},
                    {3, "Spring MVC and WebFlow"},
                    {4, "JavaEE and WildFly Module 1 : Getting Started"},
                    {5, "Hibernate and JPA"},
                    {6, "Java Web Development Second Edition: Module 1"},
                    {7, "Java Fundamentals"},
                    {8, "NoSQL Databases"},
                    {9, "Java Advanced Topics"},
                    {10, "Docker for Java Developers"},
                    {11, "Java Web Development Second Edition: Module 2"},
                    {12, "HTML5 and Responsive CSS for Developers"},
                    {13, "Git"},
                    {14, "Spring Boot"},
                    {15, "Groovy Programming"},
                    {16, "Java Build Tools"},
                    {17, "Hadoop for Java Developers"},
                    {18, "Cloud Deployment with AWS"},
                    {19, "Docker Module 2 for Java Developers"},
                    {20, "Going Further with Android"},
                    {21, "Test Driven Development"},
                    {22, "Introduction to Android"},
                    {23, "Java Web Development"},
                    {24, "Spring Security Module 3"},
                    {25, "Java Messaging with JMS and MDB"},
                    {26, "Spring Remoting and Webservices"},
                    {27, "Thymeleaf"},
                    {28, "Spring Security Module 2: OAuth2 and REST"},
                    {29, "JavaEE and WildFly Module 2: Webservices"},
                    {30, "Spring Security Core Concepts"},
                    {31, "JavaEE and Wildfly Module 3: Messaging"},
                    {32, "JavaEE"},
                    {33, "Microservice Deployment"},
                    {34, "Securing a VPC"},
                    {35, "WTP Plugins for Eclipse"},
                    {36, "Spark for Java Developers"},
                    {37, "JavaEE and Wildfly Module 4: JSF"},
                    {38, "Kubernetes Microservices Module 1"},
                    {39, "Kotlin with Spring Boot"},
                    {40, "Kubernetes Microservices Module 2"},
                    {41, "Spark Module 2 SparkSQL and DataFrames"},
                    {42, "Spark Module 3 Machine Learning SparkML"}})
            .collect(Collectors.toMap(it -> (Integer) it[0], it -> (String) it[1]));

    public static void main(String[] args) throws InterruptedException, FileNotFoundException {
        var props = new Properties();
        props.put("bootstrap.servers", "localhost:9092");
        props.put("acks", "all");
        props.put("retries", 0);
        props.put("batch.size", 16384);
        props.put("linger.ms", 1);
        props.put("buffer.memory", 33554432);
        props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");

        // The input file contains a full day (24 hours) of viewing figures. We'll speed this up
        // by 24x to give a one-hour simulation....
        try (var producer = new KafkaProducer<>(props);
             var scanner = new Scanner(new FileReader("src/main/resources/streaming/final_viewing_figures"))) {
            int milliseconds = 0;
            while (scanner.hasNextLine()) {
                var input = scanner.nextLine().split(",");
                var timestamp = Integer.parseInt(input[0]);
                var courseKey = Integer.valueOf(input[1]);

                // Wait until this event is due...
                while (milliseconds < timestamp) {
                    milliseconds++;
                    if (milliseconds % 24 == 0) {
                        Thread.sleep(100);
                    }
                }

                var courseName = COURSE_KEYS.get(courseKey);
                producer.send(new ProducerRecord<>("viewrecords", courseName));
            }
        }
    }
}
