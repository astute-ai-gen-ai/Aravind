import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
import psutil
import time
import pandas as pd
import threading

# Load environment variables from .env file
load_dotenv()

# Retrieve the API token from the environment variable
sec_key = "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"

if not sec_key:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the environment variables.")

# Set the API token as an environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = sec_key

# Define the repository ID and initialize the HuggingFaceEndpoint
repo_id = "microsoft/Phi-3-mini-4k-instruct"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=sec_key, timeout=60)

def fetch_website_content(url):
    """Fetch content from the given URL."""
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join(p.get_text() for p in soup.find_all('p'))
        return text
    else:
        raise ValueError(f"Failed to retrieve content from {url}, status code: {response.status_code}")

# Streamlit UI
st.header("Anna")
with st.sidebar:
    st.title("Web URLs")
    urls = st.text_area("Enter the URLs from which information has to be extracted (one per line)").splitlines()

# Extract the text
if urls:
    combined_content = ""
    url_chunks_map = {}
    chunk_id = 0

    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    for url in urls:
        if url.strip():
            content = fetch_website_content(url.strip())
            chunks = text_splitter.split_text(content)
            for chunk in chunks:
                url_chunks_map[chunk_id] = {'url': url.strip(), 'content': chunk}
                chunk_id += 1

    # Generating embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Creating vector store - FAISS
    vector_store = FAISS.from_texts([chunk['content'] for chunk in url_chunks_map.values()], embeddings)

    # Get user question
    user_question = st.text_input("Type your question here")

    # Do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        
        # Find the relevant URL
        relevant_url = None
        for doc in match:
            for chunk_id, chunk in url_chunks_map.items():
                if doc.page_content == chunk['content']:
                    relevant_url = chunk['url']
                    break
            if relevant_url:
                break

        # Output results
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)
        
        # Display the response
        st.write(response)
        
        # Display the relevant source URL at the bottom
        if relevant_url:
            st.write("\n\n**Source of the content:**")
            st.write(f"- {relevant_url}")

# Function to log resource usage
def log_resource_usage(interval=10):
    """Log the resource usage at the specified interval (in seconds)."""
    usage_data = {'time': [], 'cpu_usage': [], 'memory_usage': []}
    while True:
        # Get the current CPU and memory usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent

        # Append data
        usage_data['time'].append(time.time())
        usage_data['cpu_usage'].append(cpu_usage)
        usage_data['memory_usage'].append(memory_usage)

        # Create a DataFrame and update the line chart
        df = pd.DataFrame(usage_data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        st.session_state['usage_df'] = df

        # Wait for the next interval
        time.sleep(interval)

# Initialize session state for usage data
if 'usage_df' not in st.session_state:
    st.session_state['usage_df'] = pd.DataFrame(columns=['time', 'cpu_usage', 'memory_usage'])

# Start logging resource usage in a separate thread
threading.Thread(target=log_resource_usage, daemon=True).start()

# Display dynamic graphs
st.line_chart(st.session_state['usage_df'].set_index('time'))