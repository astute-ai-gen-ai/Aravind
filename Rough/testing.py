import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Suppress specific category of warnings, e.g., DeprecationWarning
warnings.filterwarnings('ignore', category=DeprecationWarning)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup

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

urls = input("Enter the URLs from which information has to be extracted (comma-separated) : ").split(',')
urls = [url.strip() for url in urls]  # Remove any leading/trailing whitespace

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
    user_question = input("Type your question here : ")

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
        print(response)
        
        # Display the relevant source URL at the bottom
        if relevant_url:
            print("\n\n**Source of the content:**")
            print(f"- {relevant_url}")