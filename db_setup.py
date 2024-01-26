import os
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from dotenv import load_dotenv
load_dotenv()

def init_vectordb():
    HOST = 'localhost'
    PORT = 19530

    connections.connect(host=HOST, port=PORT, secure=False)

    collection_name = 'book_covers_collection'

    if utility.has_collection(collection_name):
        print('Dropping existing collection "%s"' % collection_name)
        utility.drop_collection(collection_name)
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name='title', dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name='image_embedding', dtype=DataType.FLOAT_VECTOR, dim=512)
    ]
    
    print('Creating collection and index for "%s"' % collection_name)
    schema = CollectionSchema(fields=fields)
    collection = Collection(name=collection_name, schema=schema)
    # Create an IVF_FLAT index for collection.
    index_params = {
        'metric_type':'L2',
        'index_type':"IVF_FLAT",
        'params':{"nlist":768}
    }
    collection.create_index(field_name="image_embedding", index_params=index_params)
    collection.load()
    return collection

if __name__ == '__main__':
    init_vectordb()