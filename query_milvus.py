import os, os.path
import pandas as pd
import requests
import os
import gzip
import json
import numpy as np

import requests

from pymilvus import connections, Collection, utility


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


import os


collection_name = 'book_covers_collection'

connections.connect(host='localhost', port='19530', secure=False)
if utility.has_collection(collection_name):
    print('Collection exists "%s"' % collection_name)
else:
    print('Collection does not exist')



def get_text_embedding(text):
  if text:
    model_ID = "openai/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"


    model = CLIPModel.from_pretrained(model_ID).to(device)

    processor = CLIPProcessor.from_pretrained(model_ID)
    tokenizer = CLIPTokenizer.from_pretrained(model_ID)

    try:
      inputs = tokenizer(text, return_tensors = "pt")
      text_embeddings = model.get_text_features(**inputs)
      embedding_as_np = text_embeddings.cpu().detach().numpy()
    except:
      print('error')
      print(text)
      return None
    return embedding_as_np
  print('null topic')
  print(text)



def data_querying(input_text):
    print('Searching in vector DB ...')
    search_terms = [input_text]
    search_data = get_text_embedding(input_text)[0]
    print(search_data)
    collection = Collection(collection_name)
    response = collection.search(
        data=[search_data],
        anns_field="image_embedding",
        param={},
        limit = 3,
        output_fields=['title']
    )
    
    books = ''
    for _, hits in enumerate(response):
        for hit in hits:
            books += hit.entity.get('title') + '\n\n'

    return books

    

if __name__ == '__main__':
    input_text = 'a cozy mystery with a woman amateur sleuth'
    matches = data_querying(input_text)
    print(matches)