# langchain/vectorstore.py
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Function to create a vector store
def create_vector_store(splits):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(persist_directory='./vector_db', documents=splits, embedding=embeddings)

# Define the retriever class
class SimpleRetriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def get_relevant_documents(self, query):
        return self.vectorstore.similarity_search(query)