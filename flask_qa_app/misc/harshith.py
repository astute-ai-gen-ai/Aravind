import nltk
nltk.download('punkt')

from huggingface_hub import login
login(token="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")

from langchain_huggingface import HuggingFaceEndpoint
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from bs4 import BeautifulSoup
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import fitz  # PyMuPDF

# Initialize the language model and embeddings
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, model_kwargs={"max_length": 128})
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to fetch website content
def fetch_website_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = ' '.join(p.get_text() for p in soup.find_all('p'))
    return Document(page_content=text, metadata={"source": url})

# Function to fetch content from a PDF
def fetch_pdf_content(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return Document(page_content=text, metadata={"source": file_path})

# List of URLs and PDF paths to load
resources = [
    "The ROI of AI - ASTUTE.pdf",
    "Astute Digital Integration.pdf",
    "Astute Machine Learning.pdf",
    "Astute SaaS and AI.pdf"
]

# Load documents from URLs and PDFs
documents = []
for resource in resources:
    if resource.startswith("http"):
        documents.append(fetch_website_content(resource))
    elif resource.endswith(".pdf"):
        documents.append(fetch_pdf_content(resource))

# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# Store document vectors using Chroma
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Define the retriever class
class SimpleRetriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def get_relevant_documents(self, query):
        # Retrieve relevant documents from the vectorstore
        return self.vectorstore.similarity_search(query)

retriever = SimpleRetriever(vectorstore)

# Create a prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Given the context: {context}, answer the question: {question}"
)

# Function to format documents
def format_docs(docs):
    return " ".join([doc.page_content for doc in docs])

# Create a custom output parser
class CustomStrOutputParser:
    def __call__(self, output):
        return {"answer": output}

# Create the RAG chain
def rag_chain(context, question):
    formatted_context = format_docs(context)
    prompt = prompt_template.format(context=formatted_context, question=question)
    response = llm(prompt)
    return CustomStrOutputParser()(response)

# Function to invoke the RAG chain with history
def invoke_with_history(query, chat_history):
    # Update history with the new query
    chat_history.append({"role": "user", "content": query})

    # Retrieve relevant documents
    relevant_docs = retriever.get_relevant_documents(query)

    # Format the context with history
    history_context = "\n".join([f"{entry['role']}: {entry['content']}" for entry in chat_history])
    context_with_history = [Document(page_content=history_context, metadata={})] + relevant_docs

    # Invoke the chain
    result = rag_chain(context_with_history, query)

    # Update history with the model's response
    chat_history.append({"role": "assistant", "content": result["answer"]})

    # Extract sources
    sources = set(doc.metadata["source"] for doc in relevant_docs)

    return result["answer"], sources, chat_history

# Function to clear chat history
def clear_chat_history():
    global chat_history
    chat_history = []
chat_history = []

# Function to ask a single query
def ask_question(query):
    global chat_history
    answer, sources, chat_history = invoke_with_history(query, chat_history)
    print(f"Query: {query}")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")
    print(f"Chat History: {chat_history}")
    print()
clear_chat_history()
ask_question("What is the primary goal of AstuteAI?")