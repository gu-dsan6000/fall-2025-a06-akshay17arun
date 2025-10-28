import os
import sys
import time
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    regexp_extract, col, desc
)
from pyspark.sql import functions as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


def create_spark_session(master_url):
    
    spark = (
        SparkSession.builder
        .appName("Problem1_LogLevelDistribution")
        .master(master_url)
        
        .config("spark.jars.packages", 
                "org.apache.hadoop:hadoop-aws:3.3.4,"
                "com.amazonaws:aws-java-sdk-bundle:1.12.262")
        
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        
        .config("spark.executor.cores", "2")
        .config("spark.cores.max", "6")
        
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", 
                "com.amazonaws.auth.InstanceProfileCredentialsProvider")
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
        
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        
        .getOrCreate()
    )
    
    logger.info("Spark session created successfully for cluster execution")
    return spark


def analyze_log_levels(spark, s3_path, output_dir):
    
    logger.info("Starting log level distribution analysis")
    print("\n" + "=" * 80)
    print("Problem 1: Log Level Distribution Analysis")
    print("=" * 80)
    print(f"Master URL: {spark.sparkContext.master}")
    print(f"Data Source: {s3_path}")
    
    overall_start = time.time()
    
    logger.info(f"Reading log files from S3: {s3_path}")
    print("\nReading log files from S3...")
    print("⏳ This may take a few minutes as data is loaded from S3...")
    
    read_start = time.time()
    
    log_pattern = f"{s3_path}application_*/*.log"
    
    print(f"Using pattern: {log_pattern}")
    logger.info(f"Using file pattern: {log_pattern}")
    
    logs_df = spark.read.text(log_pattern)
    
    read_time = time.time() - read_start
    logger.info(f"Data loaded in {read_time:.2f} seconds")
    print(f"✅ Data loaded in {read_time:.2f} seconds")
    
    print("\nDataset Schema:")
    logs_df.printSchema()
    
    logger.info("Caching dataframe for multiple operations")
    logs_df.cache()
    
    print("\nCounting total log lines...")
    total_lines = logs_df.count()
    logger.info(f"Total log lines: {total_lines:,}")
    print(f"✅ Total log lines: {total_lines:,}")
    
    print("\n" + "=" * 80)
    print("STEP 1: Parsing Log Levels")
    print("=" * 80)
    logger.info("Parsing log levels from log entries")
    
    parse_start = time.time()
    
    parsed_df = logs_df.withColumn(
        "log_level",
        regexp_extract(col("value"), r'(INFO|WARN|ERROR|DEBUG)', 1)
    )
    

    parsed_df = parsed_df.filter(col("log_level") != "")
    parsed_df.cache()
    
    total_with_levels = parsed_df.count()
    parse_time = time.time() - parse_start
    
    logger.info(f"Parsed {total_with_levels:,} lines with log levels in {parse_time:.2f} seconds")
    print(f"✅ Found {total_with_levels:,} lines with log levels")
    print(f"   Execution time: {parse_time:.2f} seconds")
    
    print("\n" + "=" * 80)
    print("STEP 2: Counting Log Level Distribution")
    print("=" * 80)
    logger.info("Computing log level counts")
    
    count_start = time.time()
    
    counts_df = parsed_df.groupBy("log_level") \
        .agg(F.count("*").alias("count")) \
        .orderBy(desc("count"))
    
    count_time = time.time() - count_start
    logger.info(f"Counts computed in {count_time:.2f} seconds")
    
    print("\nLog Level Distribution:")
    counts_df.show(truncate=False)
    print(f"Execution time: {count_time:.2f} seconds")
    
    counts_df.toPandas().to_csv("problem1_counts.csv", index=False)
    logger.info(f"Saved log level counts to problem1_counts.csv")
    print(f"✅ Saved: problem1_counts.csv")
    
    print("\n" + "=" * 80)
    print("STEP 3: Sampling Random Log Entries")
    print("=" * 80)
    logger.info("Sampling random log entries")
    
    sample_start = time.time()
    
    sample_df = parsed_df.select(
        col("value").alias("log_entry"),
        col("log_level")
    ).sample(False, 0.01, seed=42).limit(10)
    
    sample_time = time.time() - sample_start
    logger.info(f"Sampled 10 entries in {sample_time:.2f} seconds")
    
    print("\nSample Log Entries:")
    sample_df.show(10, truncate=80)
    print(f"Execution time: {sample_time:.2f} seconds")
    
    sample_df.toPandas().to_csv("problem1_sample.csv", index=False)
    logger.info(f"Saved sample entries to problem1_sample.csv")
    print(f"✅ Saved: problem1_sample.csv")
    
    print("\n" + "=" * 80)
    print("STEP 4: Computing Summary Statistics")
    print("=" * 80)
    logger.info("Computing summary statistics")
    
    summary_start = time.time()
    
    counts_list = counts_df.collect()
    unique_levels = len(counts_list)
    
    summary_time = time.time() - summary_start
    logger.info(f"Summary computed in {summary_time:.2f} seconds")
    
    with open("problem1_summary.txt", 'w') as f:
        f.write(f"Total log lines processed: {total_lines:,}\n")
        f.write(f"Total lines with log levels: {total_with_levels:,}\n")
        f.write(f"Unique log levels found: {unique_levels}\n")
        f.write("\n")
        f.write("Log level distribution:\n")
        
        for row in counts_list:
            log_level = row['log_level']
            level_count = row['count']
            percentage = (level_count / total_with_levels) * 100
            f.write(f"  {log_level:<6}: {level_count:>10,} ({percentage:>5.2f}%)\n")
    
    logger.info(f"Saved summary to problem1_sample.csv")
    print(f"✅ Saved: problem1_sample.csv")
    
    print("\nSummary Statistics:")
    print(f"  Total log lines processed: {total_lines:,}")
    print(f"  Total lines with log levels: {total_with_levels:,}")
    print(f"  Unique log levels found: {unique_levels}")
    print("\n  Log level distribution:")
    for row in counts_list:
        log_level = row['log_level']
        level_count = row['count']
        percentage = (level_count / total_with_levels) * 100
        print(f"    {log_level:<6}: {level_count:>10,} ({percentage:>5.2f}%)")
    
    logs_df.unpersist()
    parsed_df.unpersist()
    
    total_time = time.time() - overall_start
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nExecution Times:")
    print(f"  Data loading: {read_time:.2f} seconds")
    print(f"  Log level parsing: {parse_time:.2f} seconds")
    print(f"  Count aggregation: {count_time:.2f} seconds")
    print(f"  Sampling: {sample_time:.2f} seconds")
    print(f"  Summary generation: {summary_time:.2f} seconds")
    print(f"  Total: {total_time:.2f} seconds")
    
    print(f"\nGenerated files in {output_dir}/:")
    print("  - problem1_counts.csv")
    print("  - problem1_sample.csv")
    print("  - problem1_summary.txt")
    
    logger.info(f"All analyses completed successfully in {total_time:.2f} seconds")
    
    return True


def main():
    
    logger.info("Starting Problem 1: Log Level Distribution Analysis")
    print("=" * 80)
    print("PROBLEM 1: LOG LEVEL DISTRIBUTION ANALYSIS")
    print("Spark Cluster Logs - 2015-2017")
    print("=" * 80)
    
    if len(sys.argv) > 1:
        master_url = sys.argv[1]
    else:
        master_private_ip = os.getenv("MASTER_PRIVATE_IP")
        if master_private_ip:
            master_url = f"spark://{master_private_ip}:7077"
        else:
            print("❌ Error: Master URL not provided")
            print("Usage: python problem1.py spark://MASTER_IP:7077")
            print("   or: export MASTER_PRIVATE_IP=xxx.xxx.xxx.xxx")
            return 1
    
    print(f"Connecting to Spark Master at: {master_url}")
    logger.info(f"Using Spark master URL: {master_url}")
    
    NET_ID = "aa2627" 
    S3_BUCKET = f"{NET_ID}-assignment-spark-cluster-logs"
    S3_PATH = f"s3a://{S3_BUCKET}/data/"
    
    print(f"\n⚠️  IMPORTANT: Make sure you've updated NET_ID in the script!")
    print(f"Current S3 bucket: {S3_BUCKET}")
    print(f"S3 path: {S3_PATH}")
    
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    overall_start = time.time()
    
    logger.info("Initializing Spark session for cluster execution")
    spark = create_spark_session(master_url)
    
    try:
        logger.info("Starting log level analysis")
        success = analyze_log_levels(spark, S3_PATH, output_dir)
        logger.info("Log level analysis completed successfully")
    except Exception as e:
        logger.exception(f"Error occurred during analysis: {str(e)}")
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        success = False
    
    logger.info("Stopping Spark session")
    spark.stop()
    
    overall_end = time.time()
    total_time = overall_end - overall_start
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    
    print("\n" + "=" * 80)
    if success:
        print("✅ PROBLEM 1 COMPLETED SUCCESSFULLY!")
        print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"\nFiles created in {output_dir}/:")
        print("  - problem1_counts.csv")
        print("  - problem1_sample.csv")
        print("  - problem1_summary.txt")
        print("\nNext steps:")
        print("  1. Review the output files")
        print("  2. Take screenshots of the Spark Web UI")
        print("  3. Move on to Problem 2 or clean up: ./cleanup-spark-cluster.sh")
    else:
        print("❌ Problem 1 analysis failed - check error messages above")
    print("=" * 80)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())