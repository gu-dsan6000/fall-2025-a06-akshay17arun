import os
import sys
import time
import logging
import argparse
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    regexp_extract, col, min as spark_min, max as spark_max, 
    count as spark_count, to_timestamp, input_file_name
)
from pyspark.sql import functions as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


def create_spark_session(master_url):
    
    spark = (
        SparkSession.builder
        .appName("Problem2_ClusterUsageAnalysis")
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


def analyze_cluster_usage(spark, s3_path, output_dir):
    
    logger.info("Starting cluster usage analysis")
    print("\n" + "=" * 80)
    print("Problem 2: Cluster Usage Analysis")
    print("=" * 80)
    print(f"Master URL: {spark.sparkContext.master}")
    print(f"Data Source: {s3_path}")
    print("⚠️  This analysis takes approximately 10-20 minutes")
    
    overall_start = time.time()
    
    logger.info(f"Reading log files from S3: {s3_path}")
    print("\nReading log files from S3...")
    print("⏳ This may take several minutes as data is loaded from S3...")
    
    read_start = time.time()
    
    logs_df = spark.read.option("recursiveFileLookup", "true").text(s3_path)
    
    logs_df = logs_df.withColumn("file_path", input_file_name())
    
    read_time = time.time() - read_start
    logger.info(f"Data loaded in {read_time:.2f} seconds")
    print(f"✅ Data loaded in {read_time:.2f} seconds")
    
    logger.info("Caching dataframe for multiple operations")
    logs_df.cache()
    
    total_lines = logs_df.count()
    logger.info(f"Total log lines: {total_lines:,}")
    print(f"✅ Total log lines: {total_lines:,}")
    
    print("\n" + "=" * 80)
    print("STEP 1: Extracting Application and Cluster IDs")
    print("=" * 80)
    logger.info("Extracting application and cluster IDs from file paths")
    
    extract_start = time.time()
    
    logs_with_ids = logs_df.withColumn(
        "application_id",
        regexp_extract(col("file_path"), r"application_(\d+_\d+)", 0)
    )
    
    logs_with_ids = logs_with_ids.withColumn(
        "cluster_id",
        regexp_extract(col("application_id"), r"application_(\d+)_", 1)
    )
    
    logs_with_ids = logs_with_ids.withColumn(
        "app_number",
        regexp_extract(col("application_id"), r"application_\d+_(\d+)", 1)
    )
    
    logs_with_ids = logs_with_ids.filter(col("application_id") != "")
    logs_with_ids.cache()
    
    extract_time = time.time() - extract_start
    logger.info(f"IDs extracted in {extract_time:.2f} seconds")
    print(f"✅ Application and cluster IDs extracted in {extract_time:.2f} seconds")

    print("\nSample data with IDs:")
    logs_with_ids.select("application_id", "cluster_id", "app_number").distinct().show(5, truncate=False)
    
    print("\n" + "=" * 80)
    print("STEP 2: Parsing Timestamps from Log Entries")
    print("=" * 80)
    logger.info("Parsing timestamps from log entries")
    
    parse_start = time.time()
    
    logs_with_time = logs_with_ids.withColumn(
        "timestamp_str",
        regexp_extract(col("value"), r"(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})", 1)
    )
    
    # CRITICAL FIX: Filter out empty timestamp strings BEFORE conversion
    # This prevents "CANNOT_PARSE_TIMESTAMP" errors on empty strings
    logs_with_time = logs_with_time.filter(col("timestamp_str") != "")
    
    logs_with_time = logs_with_time.withColumn(
        "timestamp",
        to_timestamp(col("timestamp_str"), "yy/MM/dd HH:mm:ss")
    )
    
    logs_with_time = logs_with_time.filter(col("timestamp").isNotNull())
    logs_with_time.cache()
    
    parse_time = time.time() - parse_start
    logger.info(f"Timestamps parsed in {parse_time:.2f} seconds")
    print(f"✅ Timestamps parsed in {parse_time:.2f} seconds")
    
    print("\n" + "=" * 80)
    print("STEP 3: Computing Application Timeline")
    print("=" * 80)
    logger.info("Computing application start and end times")
    
    timeline_start = time.time()
    
    app_timeline = logs_with_time.groupBy("cluster_id", "application_id", "app_number").agg(
        spark_min("timestamp").alias("start_time"),
        spark_max("timestamp").alias("end_time")
    ).orderBy("cluster_id", "app_number")
    
    app_timeline.cache()
    timeline_count = app_timeline.count()
    
    timeline_time = time.time() - timeline_start
    logger.info(f"Timeline computed in {timeline_time:.2f} seconds")
    print(f"✅ Application timeline computed: {timeline_count} applications")
    print(f"   Execution time: {timeline_time:.2f} seconds")
    
    print("\nSample application timeline:")
    app_timeline.show(10, truncate=False)
    
    output_file = os.path.join(output_dir, "problem2_timeline.csv")
    app_timeline.toPandas().to_csv(output_file, index=False)
    logger.info(f"Saved timeline to {output_file}")
    print(f"✅ Saved: {output_file}")
    
    print("\n" + "=" * 80)
    print("STEP 4: Computing Cluster Summary Statistics")
    print("=" * 80)
    logger.info("Computing cluster-level statistics")
    
    summary_start = time.time()
    
    cluster_summary = app_timeline.groupBy("cluster_id").agg(
        spark_count("application_id").alias("num_applications"),
        spark_min("start_time").alias("cluster_first_app"),
        spark_max("end_time").alias("cluster_last_app")
    ).orderBy(col("num_applications").desc())
    
    cluster_summary.cache()
    
    summary_time = time.time() - summary_start
    logger.info(f"Cluster summary computed in {summary_time:.2f} seconds")
    
    print("\nCluster Summary:")
    cluster_summary.show(truncate=False)
    print(f"Execution time: {summary_time:.2f} seconds")
    
    output_file = os.path.join(output_dir, "problem2_cluster_summary.csv")
    cluster_summary.toPandas().to_csv(output_file, index=False)
    logger.info(f"Saved cluster summary to {output_file}")
    print(f"✅ Saved: {output_file}")
    
    print("\n" + "=" * 80)
    print("STEP 5: Generating Overall Statistics")
    print("=" * 80)
    logger.info("Generating overall statistics")
    
    stats_start = time.time()
    
    cluster_list = cluster_summary.collect()
    total_clusters = len(cluster_list)
    total_apps = sum(row['num_applications'] for row in cluster_list)
    avg_apps_per_cluster = total_apps / total_clusters if total_clusters > 0 else 0
    
    stats_time = time.time() - stats_start
    logger.info(f"Statistics generated in {stats_time:.2f} seconds")
    
    output_file = os.path.join(output_dir, "problem2_stats.txt")
    with open(output_file, 'w') as f:
        f.write(f"Total unique clusters: {total_clusters}\n")
        f.write(f"Total applications: {total_apps}\n")
        f.write(f"Average applications per cluster: {avg_apps_per_cluster:.2f}\n")
        f.write("\n")
        f.write("Most heavily used clusters:\n")
        
        for row in cluster_list:
            cluster_id = row['cluster_id']
            num_apps = row['num_applications']
            f.write(f"  Cluster {cluster_id}: {num_apps} applications\n")
    
    logger.info(f"Saved statistics to {output_file}")
    print(f"✅ Saved: {output_file}")
    
    print("\nOverall Statistics:")
    print(f"  Total unique clusters: {total_clusters}")
    print(f"  Total applications: {total_apps}")
    print(f"  Average applications per cluster: {avg_apps_per_cluster:.2f}")
    print("\n  Most heavily used clusters:")
    for row in cluster_list:
        cluster_id = row['cluster_id']
        num_apps = row['num_applications']
        print(f"    Cluster {cluster_id}: {num_apps} applications")
    
    logs_df.unpersist()
    logs_with_ids.unpersist()
    logs_with_time.unpersist()
    app_timeline.unpersist()
    cluster_summary.unpersist()
    
    total_time = time.time() - overall_start
    
    print("\n" + "=" * 80)
    print("Spark Analysis Complete!")
    print("=" * 80)
    print(f"\nExecution Times:")
    print(f"  Data loading: {read_time:.2f} seconds")
    print(f"  ID extraction: {extract_time:.2f} seconds")
    print(f"  Timestamp parsing: {parse_time:.2f} seconds")
    print(f"  Timeline computation: {timeline_time:.2f} seconds")
    print(f"  Summary statistics: {summary_time:.2f} seconds")
    print(f"  Total Spark processing: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    
    logger.info(f"Spark analysis completed successfully in {total_time:.2f} seconds")
    
    return True


def generate_visualizations(output_dir):
    
    logger.info("Starting visualization generation")
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)
    
    viz_start = time.time()
    
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    print("\nReading CSV files...")
    timeline_file = os.path.join(output_dir, "problem2_timeline.csv")
    summary_file = os.path.join(output_dir, "problem2_cluster_summary.csv")
    
    timeline_df = pd.read_csv(timeline_file)
    summary_df = pd.read_csv(summary_file)
    
    timeline_df['start_time'] = pd.to_datetime(timeline_df['start_time'])
    timeline_df['end_time'] = pd.to_datetime(timeline_df['end_time'])
    timeline_df['duration_seconds'] = (timeline_df['end_time'] - timeline_df['start_time']).dt.total_seconds()
    
    logger.info(f"Loaded {len(timeline_df)} applications from {len(summary_df)} clusters")
    print(f"✅ Loaded data: {len(timeline_df)} applications, {len(summary_df)} clusters")
    
    print("\nGenerating bar chart...")
    bar_start = time.time()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    summary_sorted = summary_df.sort_values('num_applications', ascending=False)
    
    bars = ax.bar(range(len(summary_sorted)), 
                   summary_sorted['num_applications'],
                   color=sns.color_palette("husl", len(summary_sorted)))
    
    for i, (bar, val) in enumerate(zip(bars, summary_sorted['num_applications'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(int(val)), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Applications', fontsize=12, fontweight='bold')
    ax.set_title('Applications per Cluster', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(summary_sorted)))
    ax.set_xticklabels(summary_sorted['cluster_id'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "problem2_bar_chart.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    bar_time = time.time() - bar_start
    logger.info(f"Bar chart generated in {bar_time:.2f} seconds")
    print(f"✅ Saved: {output_file} ({bar_time:.2f}s)")
    
    print("\nGenerating density plot...")
    density_start = time.time()
    
    largest_cluster_id = summary_sorted.iloc[0]['cluster_id']
    largest_cluster_data = timeline_df[timeline_df['cluster_id'] == largest_cluster_id].copy()
    
    largest_cluster_data = largest_cluster_data[largest_cluster_data['duration_seconds'] > 0]
    
    n_samples = len(largest_cluster_data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(largest_cluster_data['duration_seconds'], 
            bins=30, 
            alpha=0.6, 
            color='steelblue',
            edgecolor='black',
            label='Histogram')
    
    kde = stats.gaussian_kde(largest_cluster_data['duration_seconds'])
    x_range = np.linspace(largest_cluster_data['duration_seconds'].min(), 
                          largest_cluster_data['duration_seconds'].max(), 
                          200)
    
    hist_vals, _ = np.histogram(largest_cluster_data['duration_seconds'], bins=30)
    kde_vals = kde(x_range)
    kde_scaled = kde_vals * (hist_vals.max() / kde_vals.max())
    
    ax.plot(x_range, kde_scaled, 'r-', linewidth=2, label='KDE')
    
    ax.set_xscale('log')
    
    ax.set_xlabel('Job Duration (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Job Duration Distribution for Largest Cluster\n(Cluster {largest_cluster_id}, n={n_samples})', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "problem2_density_plot.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    density_time = time.time() - density_start
    logger.info(f"Density plot generated in {density_time:.2f} seconds")
    print(f"✅ Saved: {output_file} ({density_time:.2f}s)")
    
    viz_time = time.time() - viz_start
    
    print(f"\n✅ All visualizations generated in {viz_time:.2f} seconds")
    logger.info(f"Visualization generation completed in {viz_time:.2f} seconds")
    
    return True


def main():
    
    parser = argparse.ArgumentParser(description='Problem 2: Cluster Usage Analysis')
    parser.add_argument('master_url', nargs='?', help='Spark master URL (e.g., spark://IP:7077)')
    parser.add_argument('--net-id', required=True, help='Your NET ID (e.g., aa2627)')
    parser.add_argument('--skip-spark', action='store_true', 
                       help='Skip Spark processing and only regenerate visualizations')
    
    args = parser.parse_args()
    
    logger.info("Starting Problem 2: Cluster Usage Analysis")
    print("=" * 80)
    print("PROBLEM 2: CLUSTER USAGE ANALYSIS")
    print("Spark Cluster Logs - 2015-2017")
    print("=" * 80)
    
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    overall_start = time.time()
    success = True
    
    if not args.skip_spark:
        if not args.master_url:
            master_private_ip = os.getenv("MASTER_PRIVATE_IP")
            if master_private_ip:
                master_url = f"spark://{master_private_ip}:7077"
            else:
                print("❌ Error: Master URL not provided")
                print("Usage: python problem2.py spark://MASTER_IP:7077 --net-id YOUR-NET-ID")
                print("   or: export MASTER_PRIVATE_IP=xxx.xxx.xxx.xxx")
                print("   or: python problem2.py --skip-spark --net-id YOUR-NET-ID (to skip Spark)")
                return 1
        else:
            master_url = args.master_url
        
        print(f"Connecting to Spark Master at: {master_url}")
        logger.info(f"Using Spark master URL: {master_url}")
        
        NET_ID = args.net_id
        S3_BUCKET = f"{NET_ID}-assignment-spark-cluster-logs"
        S3_PATH = f"s3a://{S3_BUCKET}/data/"
        
        print(f"\nS3 Configuration:")
        print(f"  NET ID: {NET_ID}")
        print(f"  S3 bucket: {S3_BUCKET}")
        print(f"  S3 path: {S3_PATH}")
        
        logger.info("Initializing Spark session for cluster execution")
        spark = create_spark_session(master_url)
        
        try:
            logger.info("Starting cluster usage analysis")
            success = analyze_cluster_usage(spark, S3_PATH, output_dir)
            logger.info("Cluster usage analysis completed successfully")
        except Exception as e:
            logger.exception(f"Error occurred during Spark analysis: {str(e)}")
            print(f"\n❌ Error during Spark analysis: {str(e)}")
            traceback.print_exc()
            success = False
        
        logger.info("Stopping Spark session")
        spark.stop()
    else:
        print("\n⚠️  Skipping Spark processing (--skip-spark flag set)")
        print("Using existing CSV files to regenerate visualizations...")
    
    if success:
        try:
            logger.info("Starting visualization generation")
            generate_visualizations(output_dir)
            logger.info("Visualization generation completed successfully")
        except Exception as e:
            logger.exception(f"Error occurred during visualization: {str(e)}")
            print(f"\n❌ Error during visualization: {str(e)}")
            traceback.print_exc()
            success = False
    
    overall_end = time.time()
    total_time = overall_end - overall_start
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    
    print("\n" + "=" * 80)
    if success:
        print("✅ PROBLEM 2 COMPLETED SUCCESSFULLY!")
        print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"\nFiles created in {output_dir}/:")
        print("  - problem2_timeline.csv")
        print("  - problem2_cluster_summary.csv")
        print("  - problem2_stats.txt")
        print("  - problem2_bar_chart.png")
        print("  - problem2_density_plot.png")
        print("\nNext steps:")
        print("  1. Review the output files and visualizations")
        print("  2. Take screenshots of the Spark Web UI")
        print("  3. Write your ANALYSIS.md report")
        print("  4. Clean up cluster: ./cleanup-spark-cluster.sh")
    else:
        print("❌ Problem 2 analysis failed - check error messages above")
    print("=" * 80)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())