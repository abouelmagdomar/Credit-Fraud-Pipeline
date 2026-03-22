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
 
y = df_pandas["Class"].values
 
# Scale only continuous features — hour_of_day and is_night must stay as
# integers so the dashboard can group by hour correctly.
NO_SCALE    = ("hour_of_day", "is_night")
scale_cols  = [c for c in feature_cols if c not in NO_SCALE]
noscale_cols = [c for c in feature_cols if c in NO_SCALE]
 
scaler   = SklearnScaler()
X_scaled = scaler.fit_transform(df_pandas[scale_cols].values)
 
# Combine scaled continuous cols with unscaled categorical cols
X_all            = np.hstack([X_scaled, df_pandas[noscale_cols].values])
feature_cols_ordered = scale_cols + noscale_cols
 
print(f"Before SMOTE — Class 0: {sum(y==0):,}  Class 1: {sum(y==1):,}")
 
smote = SMOTE(random_state=42, sampling_strategy=0.5)
X_resampled, y_resampled = smote.fit_resample(X_all, y)
 
print(f"After SMOTE  — Class 0: {sum(y_resampled==0):,}  Class 1: {sum(y_resampled==1):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert back to Spark and write Silver

# COMMAND ----------

pdf_resampled = pd.DataFrame(X_resampled, columns=feature_cols_ordered)
 
# SMOTE interpolates all columns as floats — snap the integer columns back.
pdf_resampled["hour_of_day"] = pdf_resampled["hour_of_day"].round().astype(int).clip(0, 23)
pdf_resampled["is_night"]    = pdf_resampled["is_night"].round().astype(int).clip(0, 1)
pdf_resampled["Class"]       = y_resampled.astype(int)
 
df_silver = spark.createDataFrame(pdf_resampled)
 
df_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
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