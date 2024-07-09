# langchain/llm_chain.py
#from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpoint
#from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from huggingface_hub import login
login(token="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")
# Initialize the language model
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, model_kwargs={"max_length": 128})

# Create a prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Answer the question: {question} based on the following context:\n{context}"
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
def invoke_with_history(query, retriever, chat_history):
    chat_history.append({"role": "user", "content": query})
    relevant_docs = retriever.get_relevant_documents(query)
    history_context = "\n".join([f"{entry['role']}: {entry['content']}" for entry in chat_history])
    context_with_history = [Document(page_content=history_context, metadata={})] + relevant_docs
    result = rag_chain(context_with_history, query)
    chat_history.append({"role": "assistant", "content": result["answer"]})
    sources = set(doc.metadata["source"] for doc in relevant_docs)
    return result["answer"], sources, chat_history