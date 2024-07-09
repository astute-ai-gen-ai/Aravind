# langchain/embed_text_chunks.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
#from langchain.schema import Document
import fitz  # PyMuPDF
import requests

# Initialize the embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to fetch content from API endpoint
def fetch_content_from_api(api_url):
    response = requests.get(api_url)
    response.raise_for_status()  # Raise an error for bad responses
    data = response.json()
    documents = []
    for item in data:
        content = f"Title: {item['title']}\nAuthor: {item['author']}\nDate: {item['date_created']}\n\n{item['content']}"
        documents.append(Document(page_content=content, metadata={"source": item["title"]}))
    return documents

# Function to fetch content from a PDF
def fetch_pdf_content(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return Document(page_content=text, metadata={"source": file_path})

# Function to fetch website content
def fetch_website_content(url):
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = ' '.join(p.get_text() for p in soup.find_all('p'))
    return Document(page_content=text, metadata={"source": url})

# Split documents into manageable chunks
def split_documents(documents):
    #from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)