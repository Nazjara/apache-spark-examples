package com.nazjara;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class Util 
{
	public static Set<String> borings = new HashSet<>();
	
	static {
		InputStream is = Util.class.getResourceAsStream("/subtitles/boringwords.txt");
		BufferedReader br = new BufferedReader(new InputStreamReader(is));
		br.lines().forEach(borings::add);
	}

	public static boolean isBoring(String word)
	{
		return borings.contains(word);
	}

	public static boolean isNotBoring(String word)
	{
		return !isBoring(word);
	}
}
