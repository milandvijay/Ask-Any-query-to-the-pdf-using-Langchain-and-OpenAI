from PyPDF2 import PdfReader 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import streamlit as st
import os 
import pickle

# Set OpenAI API Key
os.environ["OPENAI-API-KEY"] = 'write your api token here'

st.title("PDF extractor")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
submit = st.button("Process the file ")

filepath = "PDF_extracted.pkl"
if submit:
    raw_text = ''
    pdf_reader = PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    ) 
    texts = text_splitter.split_text(raw_text) 

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI-API-KEY"])
    document_search = FAISS.from_texts(texts, embeddings)
    
    # Export the file 
    with open(filepath, 'wb') as f:
        pickle.dump(document_search, f)
    
query = st.text_area("Write your input here")
if query:
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            document_search = pickle.load(f)
            llm = OpenAI(openai_api_key=os.environ["OPENAI-API-KEY"], temperature=0.6, model="gpt-3.5-turbo-instruct")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            docs = document_search.similarity_search(query)

            st.subheader("Answer")
            st.write(chain.run(input_documents=docs, question=query))
