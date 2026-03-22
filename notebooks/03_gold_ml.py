# Databricks notebook source
# Notebook: 03_gold_ml
# Layer:    Gold
# Purpose:  Train Logistic Regression and Random Forest models, evaluate with
#           correct metrics for imbalanced classification, log with MLflow,
#           and write scored predictions to a Delta table.
# Input:    main.`credit-fraud-pipeline`.silver_transactions
# Output:   Delta table -> main.`credit-fraud-pipeline`.gold_predictions
#           Serialized model -> /Volumes/main/credit-fraud-pipeline/data/gold/rf_model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports and config

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics

import mlflow
import mlflow.spark
from mlflow.models.signature import infer_signature

CATALOG    = "main"
SCHEMA     = "credit-fraud-pipeline"
TABLE_IN   = f"`{CATALOG}`.`{SCHEMA}`.silver_transactions"
TABLE_OUT  = f"`{CATALOG}`.`{SCHEMA}`.gold_predictions"
MODEL_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/data/gold/rf_model"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Silver and assemble features

# COMMAND ----------

df = spark.table(TABLE_IN)

feature_cols = [c for c in df.columns if c != "Class"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df).select("features", "Class")

print(f"Loaded {df_assembled.count():,} rows from silver")
display(df_assembled.limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train / test split

# COMMAND ----------

df_train, df_test = df_assembled.randomSplit([0.8, 0.2], seed=42)

print(f"Train rows : {df_train.count():,}")
print(f"Test rows  : {df_test.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define evaluation helper

# COMMAND ----------

def evaluate_model(predictions, model_name):
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score
    )
    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="Class",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    evaluator_pr = BinaryClassificationEvaluator(
        labelCol="Class",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )

    roc_auc = evaluator_auc.evaluate(predictions)
    pr_auc  = evaluator_pr.evaluate(predictions)

    from pyspark.ml.functions import vector_to_array
    pdf = predictions.select(
        col("Class").alias("actual"),
        col("prediction").alias("predicted"),
        vector_to_array(col("probability"))[1].alias("fraud_prob")
    ).toPandas()

    precision = precision_score(pdf["actual"], pdf["predicted"])
    recall    = recall_score(pdf["actual"], pdf["predicted"])
    f1        = f1_score(pdf["actual"], pdf["predicted"])

    print(f"\n=== {model_name} ===")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")

    return {
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "precision": precision, "recall": recall, "f1": f1
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Logistic Regression

# COMMAND ----------

mlflow.spark.autolog(disable=True)
mlflow.set_registry_uri("databricks-uc")

import os
os.environ["MLFLOW_DFS_TMP"] = "/Volumes/main/credit-fraud-pipeline/data/gold/mlflow_tmp"

with mlflow.start_run(run_name="logistic_regression"):
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="Class",
        maxIter=20,
        regParam=0.01
    )

    lr_model      = lr.fit(df_train)
    lr_preds      = lr_model.transform(df_test)
    lr_metrics    = evaluate_model(lr_preds, "Logistic Regression")

    mlflow.log_params({"maxIter": 20, "regParam": 0.01})
    mlflow.log_metrics(lr_metrics)
    mlflow.spark.log_model(lr_model, "lr_model")

    print("\nLogistic Regression run logged to MLflow")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Random Forest

# COMMAND ----------

import os
os.environ["MLFLOW_DFS_TMP"] = "/Volumes/main/credit-fraud-pipeline/data/gold/mlflow_tmp"

with mlflow.start_run(run_name="random_forest"):
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="Class",
        numTrees=100,
        maxDepth=10,
        seed=42
    )

    rf_model   = rf.fit(df_train)
    rf_preds   = rf_model.transform(df_test)
    rf_metrics = evaluate_model(rf_preds, "Random Forest")

    mlflow.log_params({"numTrees": 100, "maxDepth": 10})
    mlflow.log_metrics(rf_metrics)
    mlflow.spark.log_model(rf_model, "rf_model")

    print("\nRandom Forest run logged to MLflow")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance from Random Forest

# COMMAND ----------

feature_importance = sorted(
    zip(feature_cols, rf_model.featureImportances.toArray()),
    key=lambda x: x[1],
    reverse=True
)

print("=== Top 10 feature importances (Random Forest) ===")
for feat, imp in feature_importance[:10]:
    print(f"{feat:<10} {imp:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Gold predictions table

# COMMAND ----------

from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import round as spark_round

rf_preds_full = rf_model.transform(
    assembler.transform(spark.table(TABLE_IN)).select("features", "Class")
)

df_gold = rf_preds_full.select(
    col("Class").cast("integer").alias("actual"),
    col("prediction").cast("integer").alias("predicted"),
    vector_to_array(col("probability"))[1].alias("fraud_probability")
).withColumn("fraud_probability", spark_round(col("fraud_probability"), 4))

df_gold.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(TABLE_OUT)

print(f"Gold table written -> {TABLE_OUT}")
print(f"Total rows: {df_gold.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the Random Forest model to Volumes

# COMMAND ----------

rf_model.write().overwrite().save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export scored predictions to CSV for the dashboard

# COMMAND ----------

import pandas as pd

EXPORT_PATH = "/Volumes/main/credit-fraud-pipeline/data/gold/scored_predictions.csv"

pdf_gold   = df_gold.toPandas()
pdf_silver = spark.table(TABLE_IN).select("hour_of_day").toPandas()

pdf_gold["hour_of_day"] = pdf_silver["hour_of_day"].values

pdf_gold.to_csv(EXPORT_PATH, index=False)
print(f"Scored predictions exported -> {EXPORT_PATH}")
print(f"Columns: {list(pdf_gold.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final verification

# COMMAND ----------

df_check = spark.table(TABLE_OUT)

print("=== Prediction distribution ===")
df_check.groupBy("actual", "predicted").count().orderBy("actual", "predicted").show()

print("=== Fraud probability sample (actual fraud cases) ===")
df_check.filter(col("actual") == 1).select(
    "actual", "predicted", "fraud_probability"
).orderBy(col("fraud_probability").desc()).show(10)

print("=== Model comparison summary ===")
print(f"{'Metric':<15} {'Logistic Reg':>14} {'Random Forest':>14}")
print("-" * 45)
for metric in ["roc_auc", "pr_auc", "precision", "recall", "f1"]:
    print(f"{metric:<15} {lr_metrics[metric]:>14.4f} {rf_metrics[metric]:>14.4f}")

print("\n=== Delta history ===")
spark.sql(f"DESCRIBE HISTORY {TABLE_OUT}").select(
    "version", "timestamp", "operation"
).show(truncate=False)