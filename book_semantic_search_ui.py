import streamlit as st


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import torch
from PIL import Image
from io import BytesIO
import IPython.display
import matplotlib.pyplot as plt
from collections import OrderedDict
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

def get_text_embedding(text):
  # print(text)
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
      return None
    return embedding_as_np
  return None

# def get_top_N_books(query, data, top_K=4):
#   query_vect = get_text_embedding(query)

#   relevant_cols = ["title", "description.value", "image_url", "cos_sim"]

#   data["cos_sim"] = data["image_embeddings"].apply(lambda x: cosine_similarity(query_vect, x))# line 17
#   data["cos_sim"] = data["cos_sim"].apply(lambda x: x[0][0])

#   most_similar_articles = data.sort_values(by='cos_sim',  ascending=False)[1:top_K+1] # line 24

#   return most_similar_articles[relevant_cols].reset_index()

def get_top_N_books(query, data, top_K=4, search_criterion="text"):
  if(search_criterion.lower() == "text"):
    query_vect = get_text_embedding(query)
  else:
    query_vect = get_image_embedding(query)

  revevant_cols = ["title", "description", "image_url", "cos_sim"]

  data["cos_sim"] = data["image_embeddings"].apply(lambda x: cosine_similarity(query_vect, x))# line 17
  data["cos_sim"] = data["cos_sim"].apply(lambda x: x[0][0])

  unique_rows = data.drop_duplicates(subset=['title', 'cos_sim'])
  most_similar_articles = unique_rows.sort_values(by='cos_sim',  ascending=False)[1:top_K+1] # line 24

  return most_similar_articles[revevant_cols].reset_index()

def main():

    st.header("Semantic Book Search")
    st.text("Find exactly the book you feel like reading with natural language")

    # add some space between header and search box
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")

    
    search_col1, search_col2 = st.columns([3,1])

    with search_col1:
        input_text = st.text_input("Enter text", value="I feel like reading...", label_visibility="collapsed")

    with search_col2:
        submit_search = st.button("Show Images", type="secondary")


    col1, col2, col3, col4 = st.columns([1,2,2,2])

    with col1:
        st.write('<p style="font-size:15px; font-weight:bold; color:black;">EXAMPLES</p>', unsafe_allow_html=True)

    with col2:
        example_one = st.button("a cozy mystery with a woman amateur sleuth", type="primary")

    with col3:
        example_two = st.button("a scifi where aliens make contact with humans", type="primary")

    with col4:
        example_three = st.button("literary fiction set in new york city in the early 1900s", type="primary")

    

    st.markdown(
        """
        <style>
        button[kind="primary"] {
            background: none!important;
            border: none;
            padding: 0!important;
            color: black !important;
            cursor: pointer;
            border: none !important;
            text-decoration: underline;
            text-decoration-thickness: 1px;
            text-decoration-color: #22bfa2;
        }
        button[kind="secondary"] {
            background-color: #22bfa2;

        }
        button[kind="primary"]:hover {
            text-decoration: none;
            color: black !important;
        }
        button[kind="primary"]:focus {
            outline: none !important;
            box-shadow: none !important;
            color: black !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Button to trigger the display of images
    if (input_text and submit_search) or example_one or example_two or example_three:
        df = pd.read_pickle('https://public007-v5.s3.us-west-2.amazonaws.com/image_embeddings.pkl')
        returned_books = []
        # Display five placeholder images in a horizontal row using columns
        if example_one:
            returned_books = get_top_N_books('cozy mystery with a woman amateur sleuth', df, 15)

        elif example_two:
            returned_books = get_top_N_books('a scifi where aliens make contact with humans', df, 15)


        elif example_three:
            returned_books = get_top_N_books('literary fiction set in new york city in the early 1900s', df, 15)


        else:
            returned_books = get_top_N_books(input_text, df, 15)
        images = returned_books['image_url'].tolist()
        titles = returned_books['title'].tolist()

        idx = 0 
        for _ in range(3): 
            cols = st.columns(5) 

            with cols[0]:
                st.image(images[idx], caption=titles[idx])

            idx+=1
        
            with cols[1]:
                st.image(images[idx], caption=titles[idx])

            idx+=1

            with cols[2]:
                st.image(images[idx], caption=titles[idx])

            idx+=1

            with cols[3]:
                st.image(images[idx], caption=titles[idx])

            idx+=1

            with cols[4]:
                st.image(images[idx], caption=titles[idx])

            idx+=1
        # Set the spacing
        # spacing = 10

        # # Create two columns to display the images side by side
        # col1, col2, col3, col4, col5 = st.columns(5)

        # # Display images in columns with adjusted spacing
        # with col1:
        #     st.image(images[0], caption=titles[0])
        # st.empty()  # Add space between images
        # with col2:
        #     st.image(images[1], caption=titles[1])
        # st.empty()
        # with col3:
        #     st.image(images[2], caption=titles[2])
        # st.empty()
        # with col4:
        #     st.image(images[3], caption=titles[3])
        # st.empty()
        # with col5:
        #     st.image(images[4], caption=titles[4])

if __name__ == "__main__":
    main()
