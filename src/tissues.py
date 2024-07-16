from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pandas as pd
import json
import gcsfs


spark = SparkSession.builder.getOrCreate()

target_path = "gs://open-targets-data-releases/24.06/output/etl/parquet/targets/"
target = spark.read.parquet(target_path)

queryset = target.select("id").withColumnRenamed("id", "targetid")
hpa_data = "gs://ot-team/jroldan/hpa_data2/proteinatlas_file.json"
# Install gcsfs if not already installed

# import sys
#!{sys.executable} -m pip install gcsfs


def tissue_specific(hpa_data, queryset):
    """filter hpa_data to take the column of interests"""
    cols_of_interest = [
        "Ensembl",
        "RNA tissue distribution",
        "RNA tissue specificity",
        "Antibody",
    ]
    # reading json file
    fs = gcsfs.GCSFileSystem()
    with fs.open(hpa_data, "r") as f:
        data = json.load(f)  # Assuming the JSON file contains a single JSON object

    df = pd.DataFrame(data).filter(items=cols_of_interest)
    hpa_df = spark.createDataFrame(df)

    return hpa_df


testing = tissue_specific(hpa_data, queryset)

column_order = ["targetId", "categoryType", "categoryLabel", "categoryId"]

tissueSpecificity = (
    testing.withColumnRenamed("Ensembl", "targetId")
    .withColumn("categoryType", F.lit("tissueSpecificity"))
    .withColumn("categoryLabel", F.lit(F.col("RNA tissue specificity")))
    .withColumn("categoryId", F.lit(None))
    .select(*column_order)
)
tissueDistribution = (
    testing.withColumnRenamed("Ensembl", "targetId")
    .withColumn("categoryType", F.lit("tissueDistribution"))
    .withColumn("categoryLabel", F.lit(F.col("RNA tissue distribution")))
    .withColumn("categoryId", F.lit(None))
    .select(*column_order)
)

## union of both datasets
unionDistrSpecif = (
    (tissueSpecificity.union(tissueDistribution))
    .filter(F.col("targetId").isNotNull())
    .persist()
)

### with names and format:
targetFacetsDistrSpecif = (
    unionDistrSpecif.selectExpr(
        "categoryType as category",
        "categoryLabel as label",
        "targetId as entityIds",
        "categoryId as datasourceId",
    )
    .groupBy("category", "label", "datasourceId")
    .agg(F.collect_set("entityIds").alias("entityIds"))
    .withColumn("datasourceId", F.lit("HPA"))
)
