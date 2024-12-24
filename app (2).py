
from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA, LLMChain

# Initialize Flask app
app = Flask(__name__)

# Load documents and prepare the pipeline
loader = PyPDFDirectoryLoader("/content/")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_documents(docs)

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
vectorstore = Chroma.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = LlamaCpp(
    model_path="/content/drive/MyDrive/BioMistral-7B-Q4_K_M(3)(1)(2)(1)(1).gguf",
    temperature=0.2,
    max_tokens=2000,
    top_p=1
)

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    user_query = data.get("query", "")
    if not user_query:
        return jsonify({"answer": "Invalid query. Please provide a valid input."})

    # Use RAG Chain
    response = retriever.get_relevant_documents(user_query)  # Customize as needed
    return jsonify({"answer": response})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
