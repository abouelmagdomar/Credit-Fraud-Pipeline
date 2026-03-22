# Databricks notebook source
# Notebook: 02_silver_transform
# Layer:    Silver
# Purpose:  Feature engineering, data cleaning, and class balancing.
#           Produces a clean, ML-ready Delta table.
# Input:    main.`credit-fraud-pipeline`.bronze_transactions
# Output:   Delta table -> main.`credit-fraud-pipeline`.silver_transactions

# COMMAND ----------

# MAGIC %md
# MAGIC ## pip install

# COMMAND ----------

# MAGIC %pip install imbalanced-learn -q

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports and config

# COMMAND ----------

from pyspark.sql.functions import col, log1p, floor, when
from sklearn.preprocessing import StandardScaler as SklearnScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

CATALOG   = "main"
SCHEMA    = "credit-fraud-pipeline"
TABLE_IN  = f"`{CATALOG}`.`{SCHEMA}`.bronze_transactions"
TABLE_OUT = f"`{CATALOG}`.`{SCHEMA}`.silver_transactions"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Bronze

# COMMAND ----------

df = spark.table(TABLE_IN)
print(f"Loaded {df.count():,} rows from bronze")
df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature engineering on Time and Amount

# COMMAND ----------

df_featured = df \
    .withColumn("hour_of_day", (floor(col("Time") / 3600) % 24).cast("integer")) \
    .withColumn("is_night",    when(
                                   (col("hour_of_day") >= 22) |
                                   (col("hour_of_day") <= 5), 1
                               ).otherwise(0)) \
    .withColumn("amount_log",  log1p(col("Amount")))

display(df_featured.select("Time", "hour_of_day", "is_night", "Amount", "amount_log").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drop raw columns and define feature_cols

# COMMAND ----------

df_featured = df_featured.drop("Time", "Amount")

feature_cols = [c for c in df_featured.columns if c != "Class"]

print("Columns after feature engineering:")
print(feature_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert to Pandas, scale, and apply SMOTE

# COMMAND ----------

df_pandas = df_featured.select(*feature_cols, "Class").toPandas()

X_raw = df_pandas[feature_cols].values
y     = df_pandas["Class"].values

scaler   = SklearnScaler()
X_scaled = scaler.fit_transform(X_raw)

print(f"Before SMOTE — Class 0: {sum(y==0):,}  Class 1: {sum(y==1):,}")

smote = SMOTE(random_state=42, sampling_strategy=0.5)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print(f"After SMOTE  — Class 0: {sum(y_resampled==0):,}  Class 1: {sum(y_resampled==1):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert back to Spark and write Silver

# COMMAND ----------

pdf_resampled = pd.DataFrame(X_resampled, columns=feature_cols)
pdf_resampled["Class"] = y_resampled.astype(int)

df_silver = spark.createDataFrame(pdf_resampled)

df_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(TABLE_OUT)

print(f"Silver table written -> {TABLE_OUT}")
print(f"Total rows: {df_silver.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Silver
# MAGIC

# COMMAND ----------

df_check = spark.table(TABLE_OUT)

print("=== Class distribution after SMOTE ===")
display(df_check.groupBy("Class").count() \
    .withColumnRenamed("count", "n_rows") \
    .orderBy("Class"))

print("=== Delta history ===")
display(spark.sql(f"DESCRIBE HISTORY {TABLE_OUT}").select(
    "version", "timestamp", "operation"
))