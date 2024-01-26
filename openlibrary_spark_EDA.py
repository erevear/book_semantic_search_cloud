import os, os.path
import pandas as pd
import requests
import os
import gzip
import json
import numpy as np
import requests

from ast import literal_eval

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, explode, regexp_extract
from pyspark.sql.types import StringType, ArrayType, FloatType, BinaryType

from pyspark.sql import SparkSession

from pyspark.sql import Row

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, MapType, ArrayType
from pyspark.sql.functions import from_json, col, concat_ws, size
from pyspark.sql.functions import explode, concat


spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext
sc.setLogLevel('ERROR')


if __name__ == '__main__':
  custom_schema = StructType([
      StructField("type", StringType(), True),
      StructField("key", StringType(), True),
      StructField("revision", IntegerType(), True),
      StructField("last_modified", DateType(), True),
      StructField("json_column", StringType(), True)  # Assuming the JSON column contains key-value pairs
  ])

  df_zipped = spark.read.option("header", "true").option("delimiter", "\t").schema(custom_schema).csv("s3://semantic-book-search/input_data/ol_dump_editions_latest.txt")
  

  json_schema = MapType(StringType(), StringType())
  df_zipped = df_zipped.withColumn("json_column", from_json(df_zipped["json_column"], json_schema))


  # df_zipped.head(10)

  book_data_df = df_zipped.select(

      "json_column.title",
      "json_column.isbn_10",
      "json_column.isbn_13",
      "json_column.publish_date",
      "json_column.key",
      "json_column.subjects",
      "json_column.description",
      "json_column.genres",
      "json_column.number_of_pages",
      "json_column.languages"
  )

  array_schema = ArrayType(StringType())

  # Convert string arrays to actual arrays using from_json
  book_data_df = book_data_df.withColumn("subjects", from_json(book_data_df["subjects"], array_schema))

  book_data_df = book_data_df.filter(col("languages").like('%/languages/eng%'))

  book_data_df_w_subject = book_data_df.select("title", "isbn_10", "isbn_13", "publish_date", "key", "description", "genres", "number_of_pages", "languages", explode("subjects").alias("subject"))

  book_data_df_w_subject = book_data_df_w_subject.groupBy('subject').count()

  book_data_df_w_subject = book_data_df_w_subject.orderBy(['count'], ascending = [False])

  df_w_year = book_data_df.withColumn("year", regexp_extract("publish_date", "\\d{4}", 0))


  df_w_year = df_w_year.withColumn("year", col("year").cast(IntegerType()))

  rdd = df_w_year.rdd.map(lambda row: (row["year"], 1))

  result_rdd = rdd.reduceByKey(lambda x, y: x + y)

  result = result_rdd.map(lambda x: Row(year=x[0], count=x[1]))

  result_df = spark.createDataFrame(result)

  result_df.show()
  result_df.write.csv("s3://semantic-book-search/job_results/book_count_by_year", header=True)
  
  spark.stop()