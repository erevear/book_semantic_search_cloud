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
from pyspark.sql.functions import udf, explode
from pyspark.sql.types import StringType, ArrayType, FloatType, BinaryType

from pyspark.sql import SparkSession

import findspark

import torch
import skimage
from PIL import Image
from io import BytesIO
import IPython.display
import matplotlib.pyplot as plt

from collections import OrderedDict
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

import PIL

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, MapType, ArrayType
from pyspark.sql.functions import from_json, col, concat_ws, size, lower
from pyspark.sql.functions import explode, concat

import PIL
PIL.Image.MAX_IMAGE_PIXELS = None

from pyspark.sql import SQLContext


spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext
sc.setLogLevel('ERROR')


def get_image_url(image_id):
  if image_id:
    return f'https://covers.openlibrary.org/b/isbn/{image_id}.jpg'
  return None

def get_image(image_URL):
  if image_URL:
    try:
      response = requests.get(image_URL)
      image = Image.open(BytesIO(response.content)).convert("RGB")
      width, height = image.size
      if width == 1 and height == 1:
        return None
      return image
    except:
      print(f"Error: {image_URL}")
      return None
    return None

def get_image_embedding(image_URL):
  image = None
  embedding_as_np = []
  try:
    image = get_image(image_URL)
  except:
    print(f"Error: {image_URL}")
    return None
  if image:

    model_ID = "openai/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Save the model to device
    model = CLIPModel.from_pretrained(model_ID).to(device)
    # Get the processor
    processor = CLIPProcessor.from_pretrained(model_ID)
    # Get the tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(model_ID)
    image = processor(
        text = None,
        images = image,
        return_tensors="pt"
        )["pixel_values"].to(device)
    embedding = model.get_image_features(image)
    # convert the embeddings to numpy array

    embedding_as_np = embedding.cpu().detach().numpy().tolist()

    return embedding_as_np[0]

  return embedding_as_np


if __name__ == '__main__':

  array_schema = ArrayType(StringType())

  book_data_df = spark.read \
    .format("com.databricks.spark.csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("s3://semantic-book-search/input_data/recent_fiction_full_split")
  
  # Convert string arrays to actual arrays using from_json
  book_data_df = book_data_df.withColumn("isbn_10", from_json(book_data_df["isbn_10"], array_schema))
  book_data_df = book_data_df.withColumn("isbn_13", from_json(book_data_df["isbn_13"], array_schema))

  df_combined_isbn = book_data_df.withColumn("isbn", concat("isbn_13", "isbn_10"))

  df_exploded_isbn_col = df_combined_isbn.select("title", "publish_date", "key", "subjects", "description", "genres", "number_of_pages", "languages", explode("isbn").alias("isbn_single"))

  get_image_url_udf = udf(get_image_url, StringType())
  spark.udf.register("get_image_url", get_image_url_udf)

  get_image_embedding_udf = udf(get_image_embedding, ArrayType(FloatType()))
  spark.udf.register("get_image_embedding", get_image_embedding_udf)

  df_w_url = df_exploded_isbn_col.withColumn("image_url", get_image_url_udf(df_exploded_isbn_col["isbn_single"]))
  df_with_embeddings = df_w_url.withColumn("image_embedding", get_image_embedding_udf("image_url"))

  df_with_embeddings = df_with_embeddings.filter(size(col("image_embedding")) > 0)

  df_str_embeddings = df_with_embeddings.withColumn("image_embedding_str", concat_ws(",", col("image_embedding").cast("array<string>")))
  df_str_embeddings = df_str_embeddings.drop("image_embedding")
  
  df_str_embeddings.write.csv("s3://semantic-book-search/job_results/book_covers_with_embeddings_full_set", header=True)

  # collection_name = 'book_covers_collection'
  # collection = Collection(collection_name)
  # df_str_embeddings = df_str_embeddings.select("isbn_single", "title", "image_embeddings")
  # collection.insert(df_str_embeddings.toPandas())
  # collection.flush()