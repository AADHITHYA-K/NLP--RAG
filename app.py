import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---------------- CONFIG ---------------- #

DOCS_FOLDER = "doc"
COLLECTION_NAME = "my_rag_docs"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:3b-instruct"

CHUNK_SIZE = 700   # smaller chunks for faster retrieval
CHUNK_OVERLAP = 140
TOP_K = 4          # fewer docs retrieved for speed
SCORE_THRESHOLD = 0.6
TEMPERATURE = 0.2

st.set_page_config(
    page_title="Cryptography RAG",
    layout="wide"
)

# ---------------- HEADER ---------------- #

st.title("Cryptography & Network Security RAG System")
st.caption("Strictly answering from William Stallings (4th Edition)")
st.markdown("---")

# ---------------- INITIALIZATION ---------------- #

@st.cache_resource
def initialize_system():
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=6333
    )

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    test_vector = embeddings.embed_query("test")
    vector_dim = len(test_vector)

    try:
        client.get_collection(COLLECTION_NAME)
    except:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_dim,
                distance=Distance.COSINE,
            ),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    if client.get_collection(COLLECTION_NAME).points_count == 0:
        st.info("Embedding documents for the first time... Please wait.")

        loader = DirectoryLoader(
            DOCS_FOLDER,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
        )
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        splits = splitter.split_documents(docs)
        vector_store.add_documents(splits)

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": SCORE_THRESHOLD,
            "k": TOP_K,
        },
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=TEMPERATURE,
        num_ctx=4096,
        max_tokens=500,   # slightly reduced for faster responses
    )

    prompt_template = """
You are a senior academic expert in Cryptography and Network Security.

Answer ONLY using the provided context from:
"Cryptography and Network Security: Principles and Practice" by William Stallings (4th Edition).

Rules:
- No outside knowledge.
- No hallucination.
- If insufficient context, respond exactly with:
"The provided document sections do not contain enough information to fully answer this."

Provide:
- Structured headings
- Definitions
- Detailed explanation
- Mechanisms and classifications
- Examples if present

Context:
{context}

Question:
{question}

Comprehensive Answer:
"""

    prompt = PromptTemplate.from_template(prompt_template)

    chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(
                f"Source: {d.metadata.get('source','unknown')} | Page: {d.metadata.get('page','?')}\n{d.page_content}"
                for d in docs
            )),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


chain = initialize_system()

# ---------------- CHAT INTERFACE ---------------- #

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.subheader("Ask a Question")

query = st.text_area("Enter your question:", height=100)

col1, col2 = st.columns([1, 1])

with col1:
    generate_btn = st.button("Generate Answer", use_container_width=True)

with col2:
    clear_btn = st.button("Clear History", use_container_width=True)

if clear_btn:
    st.session_state.chat_history = []

if generate_btn and query:
    with st.status("Generating answer...", expanded=False) as status:
        answer = chain.invoke(query)
        status.update(label="Answer ready", state="complete")
    st.session_state.chat_history.append((query, answer))

if st.session_state.chat_history:
    st.markdown("## Conversation History")

    for q, a in reversed(st.session_state.chat_history):
        st.markdown("---")
        st.markdown("### Question")
        st.write(q)
        st.markdown("### Answer")
        st.write(a)
