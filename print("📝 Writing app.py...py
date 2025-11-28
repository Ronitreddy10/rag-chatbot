print("📝 Writing app.py...")
code = """
import streamlit as st
from pypdf import PdfReader
import os

# --- IMPORTS ---
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# --- CONFIG ---
st.set_page_config(page_title="DocuTalk AI", page_icon="🤖", layout="wide")

st.markdown(\"\"\"
<style>
    .stApp { background-color: #f8f9fa; }
    h1 { color: #1E88E5; font-family: 'Helvetica Neue', sans-serif; }
    .stChatMessage { background-color: white; border-radius: 15px; padding: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    .stButton>button { background-color: #1E88E5; color: white; border-radius: 8px; border: none; }
    .stButton>button:hover { background-color: #1565C0; }
</style>
\"\"\", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("Settings")
    api_key = st.text_input("Google API Key", type="password")
    language = st.selectbox("Select Language", ["English", "Hindi", "Spanish", "French", "German", "Chinese", "Japanese", "Arabic"])
    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
    process_button = st.button("🚀 Process Documents")
    st.divider()
    st.info(f"Bot will answer in: **{language}**")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, api_key):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_conversation_chain(vectorstore, api_key, language):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.3)
    
    custom_template = f"You are a helpful AI assistant. Answer based on context. IMPORTANT: Answer in {language}.\\n\\nContext: {{context}}\\n\\nQuestion: {{question}}\\n\\nHelpful Answer in {language}:"
    QA_PROMPT = PromptTemplate(template=custom_template, input_variables=["context", "question"])
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectorstore.as_retriever(), 
        memory=memory, 
        return_source_documents=True, 
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )
    return conversation_chain

if "conversation" not in st.session_state: st.session_state.conversation = None
if "messages" not in st.session_state: st.session_state.messages = []

col1, col2 = st.columns([1, 5])
with col1: st.image("https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg", width=60)
with col2: st.title("DocuTalk AI")

if process_button and api_key:
    if not pdf_docs: st.warning("⚠️ Please upload a PDF first.")
    else:
        with st.status("⚙️ Processing...", expanded=True) as status:
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks, api_key)
            if vectorstore:
                st.session_state.conversation = get_conversation_chain(vectorstore, api_key, language)
                status.update(label="✅ Ready!", state="complete", expanded=False)

elif process_button and not api_key: st.error("❌ Please enter your Google API Key.")

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

if user_question := st.chat_input(f"Ask a question in {language}..."):
    if not st.session_state.conversation: st.error("Please upload and process a document first!")
    else:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user", avatar="👤"): st.markdown(user_question)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversation({'question': f"{user_question} (Answer in {language})"})
                    bot_response = response['answer']
                    st.markdown(bot_response)
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                except Exception as e: st.error(f"Error: {e}")
"""

with open("app.py", "w") as f:
    f.write(code)