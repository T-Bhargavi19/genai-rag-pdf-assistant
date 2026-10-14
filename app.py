
import streamlit as st
import pymupdf
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# Configure the app page
st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📚"
)

# Display the app title and description
st.title("📚 PDF Q&A Assistant")
st.write(
    "Ask questions about the textbook using "
    "Retrieval-Augmented Generation (RAG)."
)

# Path to the PDF textbook
PDF_PATH = r"D:\Data\1.RAG\ISLP_website.pdf"


# Load the embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# Extract text and create chunks from PDF pages
@st.cache_data
def load_and_chunk_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    page_chunks = []

    # Extract PDF pages 12 to 18
    for page_num in range(11, 18):
        page_text = doc[page_num].get_text()
        start = 0
        chunk_size = 1000
        overlap = 200

        while start < len(page_text):
            end = start + chunk_size

            page_chunks.append({
                "page_number": page_num + 1,
                "text": page_text[start:end]
            })

            if end >= len(page_text):
                break

            start = end - overlap

    doc.close()
    return page_chunks


# Create and cache embeddings for all chunks
@st.cache_data
def create_embeddings(texts):
    return model.encode(texts)


# Load the model and PDF chunks
model = load_embedding_model()
page_chunks = load_and_chunk_pdf(PDF_PATH)

# Generate or retrieve cached embeddings
page_chunk_texts = [chunk["text"] for chunk in page_chunks]
page_chunk_embeddings = create_embeddings(page_chunk_texts)

st.success(f"PDF loaded: {len(page_chunks)} chunks created.")

# User question input
question = st.text_input("Ask a question about the PDF:")

if st.button("Ask"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        # 1. Convert the question into an embedding
        query_embedding = model.encode(question)

        # 2. Calculate cosine similarity with all chunks
        similarities = np.dot(
            page_chunk_embeddings, query_embedding
        ) / (
            np.linalg.norm(page_chunk_embeddings, axis=1)
            * np.linalg.norm(query_embedding)
        )

        # 3. Retrieve the top 3 matching chunks
        top_k = 3
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_score = float(similarities[top_indices[0]])

        st.write(f"Similarity score: {top_score:.4f}")

        # 4. Reject questions with low similarity
        if top_score < 0.30:
            st.warning(
                "I don't have enough relevant information "
                "in this PDF to answer."
            )

        else:
            retrieved_chunks = [
                page_chunks[i] for i in top_indices
            ]

            # 5. Build context with PDF page references
            context = "\n\n".join(
                f"[PDF page {chunk['page_number']}]\n"
                f"{chunk['text']}"
                for chunk in retrieved_chunks
            )

            # 6. Ask Ollama to generate an answer
            prompt = f"""
Answer the question using only the provided context.
Give a concise answer in 2-4 sentences.
Avoid repeating information.
Cite supporting pages using [PDF page X].
Do not invent facts or line numbers.
If the context is insufficient, say:
"I don't have enough relevant information in this PDF to answer."

Context:
{context}

Question:
{question}

Answer:
"""

            with st.spinner("Generating answer..."):
                response = ollama.chat(
                    model="llama3.2:3b",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

            # 7. Display the answer
            st.subheader("Answer")
            st.write(response["message"]["content"])

            # 8. Display expandable source passages
            st.subheader("Retrieved source passages")

            for chunk in retrieved_chunks:
                with st.expander(
                    f"PDF page {chunk['page_number']}"
                ):
                    st.write(chunk["text"])
