import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")



st.set_page_config(page_title="RAG PDF Chat", page_icon="📄", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>📄 Conversational RAG with PDF</h1>"
    "<p style='text-align: center; color: #306998;'>Upload PDFs and chat with their content.</p>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Configuration 🔧")
    groq_api_key = st.text_input("Groq API Key", type="password")
    session_id = st.text_input("Session ID", value="default_session")
    st.markdown("---")
    st.info("HF_TOKEN should be set in your `.env` file for embeddings.")

if groq_api_key:
    llm = ChatGroq(api_key=groq_api_key, model="gemma2-9b-it")
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            temp_path = f"./temp_{file_name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            documents.extend(docs)

        splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        split_docs = splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
        vector_store = Chroma.from_documents(split_docs, embedding=embeddings)
        retriever = vector_store.as_retriever()

        contextualize_q_system_prompt = (
            "Give a Chat History and a Latest user question "
            "which might refer context in the context in the chat history "
            "formulate a standalone question which can be understood "
            "without the chat history do not answer the question "
            "just formulate it if needed otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            st.markdown(f"### Assistant:")
            st.info(response['answer'])
            st.markdown("### Chat History")
            for msg in session_history.messages:
                if msg.type == "human":
                    st.markdown(f"**You:** {msg.content}")
                else:
                    st.markdown(f"**Assistant:** {msg.content}")
else:
    st.warning("Please enter the Groq API Key")
