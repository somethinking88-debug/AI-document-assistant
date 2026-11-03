
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Document Assistant")
st.caption("Upload a PDF and ask questions using Retrieval-Augmented Generation.")

def get_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            key = None
    return key

api_key = get_api_key()

if not api_key:
    st.warning("Please configure your OpenAI API key to continue.")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)

if uploaded_file:
    if (
        "file_name" not in st.session_state
        or st.session_state.file_name != uploaded_file.name
    ):
        try:
            reader = PdfReader(uploaded_file)
            pages = []

            for page_number, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({
                        "page_content": text,
                        "metadata": {"page": page_number + 1}
                    })

            if not pages:
                st.error("No readable text found in this PDF.")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150
            )

            documents = splitter.create_documents(
                [page["page_content"] for page in pages],
                metadatas=[page["metadata"] for page in pages]
            )

            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=api_key
            )

            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.file_name = uploaded_file.name
            st.session_state.messages = []

            st.success("PDF processed successfully!")

        except Exception as error:
            st.error(f"Could not process PDF: {error}")
            st.stop()

    st.subheader("💬 Ask your document")

    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your PDF...")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your document..."):
                try:
                    results = st.session_state.vectorstore.similarity_search(
                        question,
                        k=4
                    )

                    context = "\n\n".join(
                        f"Page {doc.metadata.get('page', 'Unknown')}:\n"
                        f"{doc.page_content}"
                        for doc in results
                    )

                    llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        temperature=0,
                        api_key=api_key
                    )

                    prompt = f"""
You are an AI document assistant.

Answer using only the provided document context.
If the answer is not in the context, say so clearly.
Do not invent facts.
Mention relevant page numbers when possible.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

                    response = llm.invoke(prompt)
                    answer = response.content

                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                    with st.expander("View retrieved sources"):
                        for doc in results:
                            st.write(
                                f"Page {doc.metadata.get('page', 'Unknown')}"
                            )
                            st.write(doc.page_content[:500])

                except Exception as error:
                    st.error(f"Something went wrong: {error}")
