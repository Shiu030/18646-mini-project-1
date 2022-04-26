import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobPriority;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class NgramCount {
	
	public static class NgramMapper extends Mapper<Object, Text, Text, Text> {
		private final static Text NUMBER_OF_WORDS = new Text("Number of words");
		
		private Text outputKey = new Text();
		private Text outputValue = new Text();
		
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String[] sixWords = new String[6];
			int count = 0;
			
			StringTokenizer itr = new StringTokenizer(value.toString());
			while (itr.hasMoreTokens()) {
				String currWord = itr.nextToken();
				sixWords[count % 6] = currWord;
				
				outputValue.set(currWord);
				
				// Output 1-gram
				context.write(NUMBER_OF_WORDS, outputValue);
				
				// Output 2-gram
				if (count > 0) {
					outputKey.set(sixWords[(count + 5) % 6]);
					context.write(outputKey, outputValue);
				}
				
				// Output 3-gram
				if (count > 1) {
					outputKey.set(sixWords[(count + 4) % 6] + " " + sixWords[(count + 5) % 6]);
					context.write(outputKey, outputValue);
				}
				
				// Increment count
				count++;
			}
		}
	}
	
	public static class NgramReducer extends Reducer<Text, Text, Text, DoubleWritable> {
		private final static String NUMBER_OF_WORDS = "Number of words";
		
		private Text outputKey = new Text();
		private DoubleWritable result = new DoubleWritable();
		
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			String keyString = key.toString();
			
			if (keyString.equals(NUMBER_OF_WORDS)) {
				// Compute 1-gram frequencies
				computeOneGramFrequency(key, values, context);
			} else {
				// Compute 2-gram and 3-gram frequencies
				computeNGramFrequencies(key, values, context);
			}
		}
		
		private void computeOneGramFrequency(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			int totalWordCount = 0;
			Map<String, Integer> wordFrequencies = new HashMap<>();
			
			for (Text val : values) {
				// Increment total word count
				totalWordCount++;
				
				// Update word frequency
				String valString = val.toString();
				wordFrequencies.put(valString, wordFrequencies.getOrDefault(valString, 0) + 1);
			}
			
			// Output 1-gram frequencies
			for (String word : wordFrequencies.keySet()) {
				double frequency = ((double)wordFrequencies.get(word)) / ((double)totalWordCount);
				
				outputKey.set(word);
				result.set(frequency);
				context.write(outputKey, result);
			}
		}
		
		private void computeNGramFrequencies(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			int wordCount = 0;
			Map<String, Integer> nGramFrequencies = new HashMap<>();
			
			for (Text val : values) {
				// Increment word count
				wordCount++;
				
				// Update n-gram frequency
				String valString = val.toString();
				nGramFrequencies.put(valString, nGramFrequencies.getOrDefault(valString, 0) + 1);
			}
			
			// Output n-gram frequencies
			String keyString = key.toString() + " ";
			
			for (String word : nGramFrequencies.keySet()) {
				double frequency = ((double)nGramFrequencies.get(word)) / ((double)wordCount);
				
				outputKey.set(keyString + word);
				result.set(frequency);
				context.write(outputKey, result);
			}
		}
	}
	
	public static class NgramPartitioner extends Partitioner<Text, Text> {
		private final static String NUMBER_OF_WORDS = "Number of words";
		
		public int getPartition(Text key, Text value, int numReduceTasks) {
			String keyString = key.toString();
			
			if (keyString.equals(NUMBER_OF_WORDS)) {
				// Send 1-gram to one reducer
				return 0;
			} else {
				// Send 2-gram and 3-gram to other reducers
				String[] keyWords = keyString.split(" ");
				return 1 + (keyWords[0].hashCode() & Integer.MAX_VALUE) % (numReduceTasks - 1);
			}
		}
	}
	
	public static class NgramHashmapMapper extends Mapper<Object, Text, Text, MapWritable> {
		private final static String NUMBER_OF_WORDS = "Number of words";
		private final static IntWritable ZERO = new IntWritable(0);
		
		private Text outputKey = new Text();
		
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String[] sixWords = new String[6];
			int count = 0;
			
			Map<String, MapWritable> contextWordCountMap = new HashMap<>();
			MapWritable outputValueMap;
			
			String currContext;
			Text t = new Text();
			IntWritable iw = new IntWritable();
			
			StringTokenizer itr = new StringTokenizer(value.toString());
			while (itr.hasMoreTokens()) {
				String currWord = itr.nextToken();
				sixWords[count % 6] = currWord;
				
				// Store 1-gram
				outputValueMap = contextWordCountMap.getOrDefault(NUMBER_OF_WORDS, new MapWritable());
				
				t.set(currWord);
				iw.set(((IntWritable)outputValueMap.getOrDefault(t, ZERO)).get() + 1);
				
				outputValueMap.put(t, iw);
				contextWordCountMap.put(NUMBER_OF_WORDS, outputValueMap);
				
				// Store 2-gram
				if (count > 0) {
					currContext = sixWords[(count + 5) % 6];
					
					outputValueMap = contextWordCountMap.getOrDefault(currContext, new MapWritable());
					
					t.set(currWord);
					iw.set(((IntWritable)outputValueMap.getOrDefault(t, ZERO)).get() + 1);
					
					outputValueMap.put(t, iw);
					contextWordCountMap.put(currContext, outputValueMap);
				}
				
				// Store 3-gram
				if (count > 1) {
					currContext = sixWords[(count + 4) % 6] + " " + sixWords[(count + 5) % 6];
					
					outputValueMap = contextWordCountMap.getOrDefault(currContext, new MapWritable());
					
					t.set(currWord);
					iw.set(((IntWritable)outputValueMap.getOrDefault(t, ZERO)).get() + 1);
					
					outputValueMap.put(t, iw);
					contextWordCountMap.put(currContext, outputValueMap);
				}
				
				// Increment count
				count++;
				
				// Clear out hashmap occasionally to prevent overflow
				if (count % 100 == 0) {
					// Output to context
					for (String contextKey : contextWordCountMap.keySet()) {
						outputKey.set(contextKey);
						context.write(outputKey, contextWordCountMap.get(contextKey));
					}
					
					contextWordCountMap = new HashMap<String, MapWritable>();
				}
			}
		}
	}
	
	public static class NgramHashmapReducer extends Reducer<Text, MapWritable, Text, DoubleWritable> {
		private final static String NUMBER_OF_WORDS = "Number of words";
		
		private Text outputKey = new Text();
		private DoubleWritable result = new DoubleWritable();
		
		public void reduce(Text key, Iterable<MapWritable> values, Context context) throws IOException, InterruptedException {
			String keyString = key.toString();
			
			if (keyString.equals(NUMBER_OF_WORDS)) {
				// Compute 1-gram frequencies
				computeOneGramFrequency(key, values, context);
			} else {
				// Compute 2-gram and 3-gram frequencies
				computeNGramFrequencies(key, values, context);
			}
		}
		
		private void computeOneGramFrequency(Text key, Iterable<MapWritable> values, Context context) throws IOException, InterruptedException {
			int totalWordCount = 0;
			Map<String, Integer> wordFrequencies = new HashMap<>();
			
			for (MapWritable val : values) {
				String wordString;
				int wordCount;
				
				for (Writable word : val.keySet()) {
					wordString = word.toString();
					wordCount = ((IntWritable) val.get(word)).get();
					
					// Update word frequency
					wordFrequencies.put(wordString, wordFrequencies.getOrDefault(wordString, 0) + wordCount);
					
					// Increment total word count
					totalWordCount += wordCount;
				}
			}
			
			// Output 1-gram frequencies
			for (String word : wordFrequencies.keySet()) {
				double frequency = ((double)wordFrequencies.get(word)) / ((double)totalWordCount);
				
				outputKey.set(word);
				result.set(frequency);
				context.write(outputKey, result);
			}
		}
		
		private void computeNGramFrequencies(Text key, Iterable<MapWritable> values, Context context) throws IOException, InterruptedException {
			int totalWordCount = 0;
			Map<String, Integer> nGramFrequencies = new HashMap<>();
			
			for (MapWritable val : values) {
				String wordString;
				int wordCount;
				
				for (Writable word : val.keySet()) {
					wordString = word.toString();
					wordCount = ((IntWritable) val.get(word)).get();
					
					// Update word frequency
					nGramFrequencies.put(wordString, nGramFrequencies.getOrDefault(wordString, 0) + wordCount);
					
					// Increment total word count
					totalWordCount += wordCount;
				}
			}
			
			// Output n-gram frequencies
			String keyString = key.toString() + " ";
			
			for (String word : nGramFrequencies.keySet()) {
				double frequency = ((double)nGramFrequencies.get(word)) / ((double)totalWordCount);
				
				outputKey.set(keyString + word);
				result.set(frequency);
				context.write(outputKey, result);
			}
		}
	}
	
	public static class NgramHashmapPartitioner extends Partitioner<Text, MapWritable> {
		private final static String NUMBER_OF_WORDS = "Number of words";
		
		public int getPartition(Text key, MapWritable value, int numReduceTasks) {
			String keyString = key.toString();
			
			if (keyString.equals(NUMBER_OF_WORDS)) {
				// Send 1-gram to one reducer
				return 0;
			} else {
				// Send 2-gram and 3-gram to other reducers
				String[] keyWords = keyString.split(" ");
				return 1 + (keyWords[0].hashCode() & Integer.MAX_VALUE) % (numReduceTasks - 1);
			}
		}
	}

	public static void main(String[] args) throws Exception {
		Configuration config = new Configuration();
		
		// Preserve state across calls
		config.setInt(Job.JVM_NUMTASKS_TORUN, -1);
		
		Job job = Job.getInstance(config, "n-gram");
		job.setJarByClass(NgramCount.class);
		job.setJobName("NgramCount");
		job.setPriority(JobPriority.VERY_HIGH);
		
		// Set input and output locations
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		
		// Set Mapper and Reducer class
		job.setMapperClass(NgramMapper.class);
		job.setReducerClass(NgramReducer.class);
//		job.setMapperClass(NgramHashmapMapper.class);
//		job.setReducerClass(NgramHashmapReducer.class);
		
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(Text.class);
//		job.setMapOutputValueClass(MapWritable.class);
		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(DoubleWritable.class);
		
		// Set Partitioner class
		job.setPartitionerClass(NgramPartitioner.class);
//		job.setPartitionerClass(NgramHashmapPartitioner.class);
		
		// Try different number of reducer tasks
		job.setNumReduceTasks(3);
		
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}
