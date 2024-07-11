
import json
from main import cleaned_pages
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from dotenv import load_dotenv
import dill as pickle
# Initialize text splitter
load_dotenv()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=100,
    length_function=len
)

# Chunk text for each page
all_chunks = []
page_indices = []
for page_num, text in enumerate(cleaned_pages):
    chunks = text_splitter.split_text(text=text)
    all_chunks.extend(chunks)
    page_indices.extend([page_num] * len(chunks))

# Generate embeddings
embeddings = OpenAIEmbeddings()
with get_openai_callback() as cb:
    vector_store = FAISS.from_texts(all_chunks, embedding=embeddings)
    print(cb)


# Save the FAISS index
vector_store.save_local("vector_store")

# Save the page indices
with open('page_indices.json', 'w') as f:
    json.dump(page_indices, f)


def query_vector_store(query_text):
  
    vector_store = FAISS.load_local("vector_store", embeddings)
    with open('page_indices.json', 'r') as f:
        page_indices = json.load(f)

    results = vector_store.similarity_search(query_text, k=5)
    
    # Display results with page numbers
    for result in results:
        chunk_text = result.page_content
        page_number = page_indices[result.metadata['id']]
        print(f"Page {page_number + 1}:")
        print(chunk_text)
        print("\n" + "="*80 + "\n")

# Example query
query_text = "Astute stool wood"
query_vector_store(query_text)
