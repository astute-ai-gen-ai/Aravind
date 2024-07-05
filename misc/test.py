import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the API token from the environment variable
sec_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not sec_key:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the environment variables.")

# Set the API token as an environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = sec_key

# Define the repository ID and initialize the HuggingFaceEndpoint
repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.3, token=sec_key, timeout=60)

# Streamlit UI
st.header("Anna")
with st.sidebar:
    st.title("Your documents")
    files = st.file_uploader("Upload your PDF files and start asking questions", type="pdf", accept_multiple_files=True)

# Initialize chat history in session state if not already present
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to process PDFs and extract text
def extract_text_from_pdfs(files):
    text = ""
    for file in files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    return text_splitter.split_text(text)

# Function to create a vector store
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embeddings)

# Extract the text and initialize the vector store if files are uploaded
vector_store = None
if files:
    text = extract_text_from_pdfs(files)
    chunks = split_text_into_chunks(text)
    vector_store = create_vector_store(chunks)

# Get user question
user_question = st.text_input("Type your question here")

# Do similarity search and generate response if a question is asked
if user_question and vector_store:
    match = vector_store.similarity_search(user_question)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=match, question=user_question)
    
    # Save the question and response in chat history
    st.session_state.chat_history.append({"question": user_question, "response": response})
    
    # Display the response
    st.write(response)

# Display the chat history
st.subheader("Chat History")
for entry in st.session_state.chat_history:
    st.write(f"**You:** {entry['question']}")
    st.write(f"**Anna:** {entry['response']}")