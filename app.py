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

app = Flask(__name__)



VectorStore = None

@app.route('/upload', methods=['POST'])
def upload_pdf():
    global VectorStore
    
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file uploaded'}), 400
    
    pdf = request.files['pdf']
    
    pdf_reader = PdfReader(pdf)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    
    embeddings = OpenAIEmbeddings()
    with get_openai_callback() as cb:
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        print(cb)
    
    return jsonify({'message': 'PDF uploaded and processed successfully'})

@app.route('/ask', methods=['POST'])
def ask_question():
    global VectorStore

    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400

  
    docs = VectorStore.similarity_search(query=query, k=3)
    llm = OpenAI()
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=query)
        print("charges:",cb)    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
