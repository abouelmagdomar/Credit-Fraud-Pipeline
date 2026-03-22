# Databricks notebook source
# Layer:    Bronze
# Purpose:  Ingest raw CSV into a Delta table with schema enforcement,
#           row-level validation, and a data quality summary.
# Input:    /Volumes/main/credit-fraud-pipeline/data/bronze/creditcard.csv
# Output:   Delta table -> main.`credit-fraud-pipeline`.bronze_transactions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports and config

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    DoubleType, IntegerType
)
from pyspark.sql.functions import col, isnan, when, count

CATALOG   = "main"
SCHEMA    = "credit-fraud-pipeline"
RAW_PATH  = "/Volumes/main/credit-fraud-pipeline/data/bronze/creditcard.csv"
TABLE_OUT = f"`{CATALOG}`.`{SCHEMA}`.bronze_transactions"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the schema explicitly

# COMMAND ----------

bronze_schema = StructType([
    StructField("Time",   DoubleType(),  nullable=False),
    StructField("V1",     DoubleType(),  nullable=False),
    StructField("V2",     DoubleType(),  nullable=False),
    StructField("V3",     DoubleType(),  nullable=False),
    StructField("V4",     DoubleType(),  nullable=False),
    StructField("V5",     DoubleType(),  nullable=False),
    StructField("V6",     DoubleType(),  nullable=False),
    StructField("V7",     DoubleType(),  nullable=False),
    StructField("V8",     DoubleType(),  nullable=False),
    StructField("V9",     DoubleType(),  nullable=False),
    StructField("V10",    DoubleType(),  nullable=False),
    StructField("V11",    DoubleType(),  nullable=False),
    StructField("V12",    DoubleType(),  nullable=False),
    StructField("V13",    DoubleType(),  nullable=False),
    StructField("V14",    DoubleType(),  nullable=False),
    StructField("V15",    DoubleType(),  nullable=False),
    StructField("V16",    DoubleType(),  nullable=False),
    StructField("V17",    DoubleType(),  nullable=False),
    StructField("V18",    DoubleType(),  nullable=False),
    StructField("V19",    DoubleType(),  nullable=False),
    StructField("V20",    DoubleType(),  nullable=False),
    StructField("V21",    DoubleType(),  nullable=False),
    StructField("V22",    DoubleType(),  nullable=False),
    StructField("V23",    DoubleType(),  nullable=False),
    StructField("V24",    DoubleType(),  nullable=False),
    StructField("V25",    DoubleType(),  nullable=False),
    StructField("V26",    DoubleType(),  nullable=False),
    StructField("V27",    DoubleType(),  nullable=False),
    StructField("V28",    DoubleType(),  nullable=False),
    StructField("Amount", DoubleType(),  nullable=False),
    StructField("Class",  IntegerType(), nullable=False),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the CSV
# MAGIC

# COMMAND ----------

df_raw = spark.read \
    .option("header", True) \
    .schema(bronze_schema) \
    .csv(RAW_PATH)

print(f"Row count : {df_raw.count():,}")
print(f"Col count : {len(df_raw.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data quality checks

# COMMAND ----------

print("=== Null / NaN check per column ===")
null_counts = df_raw.select([
    count(when(col(c).isNull() | isnan(col(c)), c)).alias(c)
    for c in df_raw.columns
])
null_counts.show(vertical=True)

print("=== Class distribution ===")
df_raw.groupBy("Class").count() \
    .withColumnRenamed("count", "n_rows") \
    .orderBy("Class") \
    .show()

print("=== Amount stats ===")
df_raw.select("Amount").summary("min", "max", "mean", "stddev").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Delta

# COMMAND ----------

df_raw.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(TABLE_OUT)

print(f"Bronze table written -> {TABLE_OUT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify with time travel

# COMMAND ----------

spark.sql(f"DESCRIBE HISTORY {TABLE_OUT}").select(
    "version", "timestamp", "operation", "operationMetrics"
).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final confirmation

# COMMAND ----------

df_check = spark.table(TABLE_OUT)
print(f"Rows in bronze_transactions : {df_check.count():,}")
df_check.printSchema()