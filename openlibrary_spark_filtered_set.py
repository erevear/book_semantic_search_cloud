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
from pyspark.sql.functions import from_json, col, concat_ws, size, lower
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

  book_data_df = book_data_df.filter(col("languages").like('%/languages/eng%'))


  book_data_df = book_data_df.filter(~lower(col("subjects")).like(f"%juvenile%"))


  book_data_df = book_data_df.filter(~lower(col("subjects")).like(f"%children%"))


  book_data_df = book_data_df.filter(lower(col("subjects")).like(f"%fiction%"))


  book_data_df = book_data_df.withColumn("number_of_pages", col("number_of_pages").cast(IntegerType()))

  book_data_df = book_data_df.filter(col("number_of_pages") >= 100)


  book_data_df = book_data_df.withColumn("year", regexp_extract("publish_date", "\\d{4}", 0))


  book_data_df = book_data_df.withColumn("year", col("year").cast(IntegerType()))

  book_data_df = book_data_df.filter(col("year") >= 2010)



  print(f'count after number of pages filter {book_data_df.count()}')

  book_data_df = book_data_df.repartition(500)
  book_data_df.write.csv("s3://semantic-book-search/input_data/recent_fiction_full_split", header=True)