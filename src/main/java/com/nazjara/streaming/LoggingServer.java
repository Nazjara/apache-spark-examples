package com.nazjara.streaming;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.ServerSocket;

public class LoggingServer {

    public static void main(String[] args) throws IOException, InterruptedException {
        try (var echoSocket = new ServerSocket(8989)) {
            var socket = echoSocket.accept();

            while (true) {
                var out = new PrintWriter(socket.getOutputStream(), true);
                var sample = Math.random();
                var level = "DEBUG";

                if (sample < 0.0001) {
                    level = "FATAL";
                } else if (sample < 0.01) {
                    level = "ERROR";
                } else if (sample < 0.1) {
                    level = "WARN";
                } else if (sample < 0.5) {
                    level = "INFO";
                }

                out.println(level + "," + new java.util.Date());
                Thread.sleep(1);
            }
        }
    }
}
