from flask import Flask, render_template, request
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
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=sec_key, timeout=60)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("files")
        user_question = request.form.get("question")

        if files and user_question:
            text = ""
            for file in files:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()

            # Break it into small chunks
            text_splitter = RecursiveCharacterTextSplitter(
                separators="\n",
                chunk_size=1000,
                chunk_overlap=150,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            # Generating embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # Creating vector store - FAISS
            vector_store = FAISS.from_texts(chunks, embeddings)

            # Do similarity search
            match = vector_store.similarity_search(user_question)

            # Output results
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=match, question=user_question)
            return render_template("index.html", response=response)

    return render_template("index.html", response=None)

if __name__ == "__main__":
    app.run(debug=True)