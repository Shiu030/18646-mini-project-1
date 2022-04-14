#!/usr/bin/bash
source ${HOME}/18646_HW3/hadoop.bashrc

## Start HADOOP & HDFS
hdfs namenode -format
${HADOOP}/sbin/start-dfs.sh
${HADOOP}/sbin/start-yarn.sh
jps

## Copy across input data to HDFS
hdfs dfs -mkdir -p /user/${USER}/ngramcount/
hdfs dfs -put /afs/andrew.cmu.edu/course/18/646/HW3/input /user/${USER}/ngramcount/input

## Compile NgramCount.java, submit as a HADOOP job and get the results
hadoop com.sun.tools.javac.Main NgramCount.java
jar cf ngramcount.jar NgramCount*.class
hadoop jar ngramcount.jar NgramCount /user/${USER}/ngramcount/input /user/${USER}/ngramcount/output
hdfs dfs -get /user/${USER}/ngramcount/output output
cat output/output/part-* > results.count
hdfs dfs -rm -r /user/${USER}/ngramcount/output

## Stop HADOOP and clean up
${HADOOP}/sbin/stop-yarn.sh
${HADOOP}/sbin/stop-dfs.sh
rm -rf /tmp/*
jps

