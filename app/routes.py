from flask import render_template, request, Blueprint
from langchainn.llm_chain import invoke_with_history
from langchainn.vectorstore import SimpleRetriever, create_vector_store
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchainn.embed_text_chunks import fetch_content_from_api, split_documents
import shutil
routes = Blueprint("routes", __name__)

@routes.route("/", methods=["GET", "POST"])
def index():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory='./vector_db', embedding_function=embeddings)
    retriever = SimpleRetriever(vectorstore)
    chat_history = []
    if request.method == "POST":
        user_question = request.form.get("question")
        if user_question:
            answer, sources, chat_history = invoke_with_history(user_question, retriever, chat_history)
            return render_template("index.html", response=answer, sources=sources)

    return render_template("index.html", response=None)

@routes.route("/refresh_vector_store", methods=["GET"])
def refresh_vector_store():
    API_URL = "https://vishnupk05.pythonanywhere.com/api/fetch_posts_txt"
    documents = fetch_content_from_api(API_URL)
    splits = split_documents(documents)
    # shutil.rmtree() to delete the vector_db folder before creating a new one. 
    # Find the correct way to update vectordb instead.
    create_vector_store(splits)
    return "Vector store refreshed!"


