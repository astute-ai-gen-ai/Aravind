# app.py
import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchainn.embed_text_chunks import fetch_pdf_content, split_documents
from langchainn.vectorstore import create_vector_store, SimpleRetriever
from langchainn.llm_chain import invoke_with_history

# Load environment variables
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)

# Load documents and create vector store
resources = [
    "uploads/The ROI of AI - ASTUTE.pdf",
    "uploads/Astute Digital Integration.pdf",
    "uploads/Astute Machine Learning.pdf",
    "uploads/Astute SaaS and AI.pdf"
]

documents = [fetch_pdf_content(resource) for resource in resources]
splits = split_documents(documents)
vectorstore = create_vector_store(splits)
retriever = SimpleRetriever(vectorstore)

# Initialize chat history
chat_history = []

# Route for the main page
@app.route("/", methods=["GET", "POST"])
def index():
    global chat_history
    if request.method == "POST":
        user_question = request.form.get("question")
        if user_question:
            answer, sources, chat_history = invoke_with_history(user_question, retriever, chat_history)
            return render_template("index.html", response=answer, sources=sources)

    return render_template("index.html", response=None)

if __name__ == "__main__":
    app.run(debug=True)