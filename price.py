import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
import os
from dotenv import load_dotenv
import json

load_dotenv()

pdf_path = 'documents/Price_List_(2).pdf'

def clean_text(text):
    unwanted_sections = [
        'info', 'backtotop', 'pillows', 'astute', 'attune', 'ayles', 'chiroform',
        'dove', 'eighty', 'two', 'entail', 'exchange', 'foster', 'inertia', 'innate',
        'jif', 'l1', 'levo', 'mantra', 'prata', 'presto', 'rainbow', 'requisite',
        'res', 'rühe', 'therapod', 'tuck', 'watson', 'you', 'zip'
    ]

    cleaned_lines = []
    lines = text.split('\n')
    
    for line in lines:
        if line == "B A C K TO TO P":
            line = line.replace(' ', '')
               
        wanted_sections = ['®', 'WATSON CLUB CHAIR', 'WATSON LOVESEAT', 'WATSON SOFA', 'WATSON BENCH'] 
        contains_unwanted = any(unwanted in line.lower() for unwanted in unwanted_sections)
        contains_wanted = any(wanted in line for wanted in wanted_sections)  
        if not contains_unwanted or contains_wanted:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

cleaned_pages = []
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        cleaned_text = clean_text(text)
        cleaned_pages.append(cleaned_text)

cleaned_text_path = './cleaned_text.txt'
with open(cleaned_text_path, 'w', encoding='utf-8') as f:
    for page_num, cleaned_page in enumerate(cleaned_pages, start=1):
        f.write(f"Page {page_num}:\n")
        f.write(cleaned_page + "\n\n")

with open(cleaned_text_path, 'r', encoding='utf-8') as f:
    text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_text(text=text)
print(len(chunks))

page_chunks = []
for page_num, page_text in enumerate(cleaned_pages, start=1):
    page_chunks.extend([{"page_num": page_num, "text": chunk} for chunk in text_splitter.split_text(text=page_text)])

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts([chunk["text"] for chunk in page_chunks], embedding=embeddings)

def get_page_chunks(page_num):
    return [chunk["text"] for chunk in page_chunks if chunk["page_num"] == page_num]

query_page_num = 123
docs = get_page_chunks(query_page_num)

# Wrap text chunks in Document objects
wrapped_docs = [Document(page_content=doc) for doc in docs]

print(wrapped_docs)

llm = OpenAI()

chain = load_qa_chain(llm=llm, chain_type="stuff")

response = chain.invoke(input_documents=wrapped_docs, question=f"Please provide all the information in a structured way.")

print(response)
