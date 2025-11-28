
import streamlit as st
from pypdf import PdfReader
import os
from dotenv import load_dotenv
load_dotenv()  # load .env if present; do NOT store secrets in repo

# --- IMPORTS ---
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from typing import List
from langchain.embeddings.base import Embeddings

# Simple local embeddings wrapper using sentence-transformers to match the
# interface expected by langchain 'Embeddings' (embed_documents/embed_query).
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]):
        # returns a list of vectors
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

    def embed_query(self, text: str):
        return self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0].tolist()

    def __call__(self, text: str):
        """Provide a callable interface for single text inputs so the
        legacy code paths that expect a callable embedding function still work.
        """
        return self.embed_query(text)

from langchain_google_genai import ChatGoogleGenerativeAI
# LangChain API compatibility imports (v0.x vs v1.x)
try:
    # v0.x style import (older langchain)
    from langchain.chains import ConversationalRetrievalChain
except Exception:
    # v1.x has modular packages; support langchain_classic fallback
    from langchain_classic.chains import ConversationalRetrievalChain

try:
    from langchain.memory import ConversationBufferMemory
except Exception:
    from langchain_classic.memory import ConversationBufferMemory

try:
    from langchain.prompts import PromptTemplate
except Exception:
    # fall back to langchain_core or langchain_classic implementations
    try:
        from langchain_core.prompts.prompt import PromptTemplate
    except Exception:
        from langchain_classic.prompts import Prompt as PromptTemplate

# --- CONFIG ---
st.set_page_config(page_title="DocuTalk AI", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    h1 { color: #1E88E5; font-family: 'Helvetica Neue', sans-serif; }
    .stChatMessage { background-color: white; border-radius: 15px; padding: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    .stButton>button { background-color: #1E88E5; color: white; border-radius: 8px; border: none; }
    .stButton>button:hover { background-color: #1565C0; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("Settings")
    # Use environment GOOGLE_API_KEY if set; otherwise show a simple password input for the API key.
    env_api_key = os.getenv('GOOGLE_API_KEY')
    if env_api_key:
        api_key = env_api_key
        st.caption("Using the API key from the environment variable GOOGLE_API_KEY")
    else:
        api_key = st.text_input("Google API Key (paste here)", type="password")

    # Auto-save to .env (developer convenience option): when a key is entered, user can choose
    # to auto-save the key into a local .env file at repo root. This writes to disk; users MUST
    # ensure that .env is excluded from version control.
    # No auto-save or additional settings: minimal sidebar (enter API key only).

    # Save/remove .env controls removed — sidebar kept minimal by request.
    language = st.selectbox("Select Language", ["English", "Hindi", "Spanish", "French", "German", "Chinese", "Japanese", "Arabic"], index=0)
    # Keep UI minimal: typed API key only. No local LLM or test buttons in simple mode.
    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
    process_button = st.button("🚀 Process Documents")
    st.divider()
    st.info(f"Bot will answer in: **{language}**")

    # Initialize and detect language change
    if "language_selected" not in st.session_state:
        st.session_state.language_selected = language
        st.session_state.language_changed = False
    elif st.session_state.language_selected != language:
        # Save the previous language and set the new one
        prev_lang = st.session_state.language_selected
        st.session_state.language_selected = language
        st.session_state.language_changed = True
        # Do not clear message history or vectorstore. We'll re-create the conversation chain later
        if st.session_state.get('vectorstore') is not None:
            st.sidebar.success(f"Language switched to {language}. Conversation will be updated to the new language if you have uploaded and processed documents.")
        else:
            st.sidebar.info(f"Language switched to {language}. No processed document available; upload a PDF to enable retrieval.")
    # Inform the user of invalid API key (if flagged)
    if st.session_state.get('invalid_api_key'):
        st.sidebar.error("The GOOGLE_API_KEY appears to be invalid. Please correct or remove it in the sidebar or set a valid key in .env.")

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
        # Use local SentenceTransformer-based embeddings to avoid Google embedding API limits
        embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None


def compute_pdf_signature(uploaded_files):
    """Compute a simple signature for the uploaded PDFs (names + sizes) to detect changes."""
    if not uploaded_files:
        return ""
    pairs = []
    for f in uploaded_files:
        try:
            size = len(f.getbuffer())
        except Exception:
            size = getattr(f, 'size', 0)
        pairs.append(f"{f.name}:{size}")
    pairs.sort()
    return "|".join(pairs)


def process_documents_if_needed(pdf_docs_list, api_key, language):
    """Auto-process the PDFs when uploaded or when explicitly requested. Returns True if processed."""
    if not pdf_docs_list:
        return False
    signature = compute_pdf_signature(pdf_docs_list)
    prev_sig = st.session_state.get('pdf_signature')
    if prev_sig == signature and st.session_state.get('vectorstore') is not None:
        # already processed
        return False
    try:
        with st.spinner("Processing uploaded documents..."):
            raw_text = get_pdf_text(pdf_docs_list)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks, api_key)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.pdf_signature = signature
                # Only create the conversation chain if API key present; otherwise user can provide it later to create the chain
                if api_key:
                    st.session_state.conversation = get_conversation_chain(vectorstore, api_key, language, existing_messages=st.session_state.get('messages', []))
                else:
                    st.session_state.conversation = None
                return True
    except Exception as e:
        st.error(f"Error auto-processing documents: {e}")
        return False
    return False

def get_conversation_chain(vectorstore, api_key, language, existing_messages=None):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.3)
        st.session_state.invalid_api_key = False
    except Exception as e:
        # Surface a helpful message for invalid/expired API keys and avoid crashing
        st.error(f"Could not initialize Google Generative AI LLM. Please check your GOOGLE_API_KEY. Error: {e}")
        st.session_state.invalid_api_key = True
        return None
    
    # Stronger language enforcement in the prompt template
    custom_template = (
        "You are a helpful AI assistant. Answer based on the context. "
        f"IMPORTANT: ONLY answer in {language}. If you cannot answer in {language}, respond with 'I cannot answer in the selected language.' "
        "Only use the specified language in every part of the response. Do not add any other language or translations.\n\n"
        "Context: {context}\n\nQuestion: {question}\n\nAnswer in " + language + ":"
    )
    QA_PROMPT = PromptTemplate(template=custom_template, input_variables=["context", "question"])
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    # If we have existing messages from the UI, populate the memory so the new chain has context
    if existing_messages:
        try:
            cm = memory.chat_memory
            for m in existing_messages:
                role = m.get('role')
                content = m.get('content')
                # Using the chat memory helpers to append past messages
                if role == 'user' and hasattr(cm, 'add_user_message'):
                    cm.add_user_message(content)
                elif role == 'assistant' and hasattr(cm, 'add_ai_message'):
                    cm.add_ai_message(content)
                elif hasattr(cm, 'add_message'):
                    cm.add_message(role, content)
        except Exception:
            # If we cannot append, ignore gracefully (memory may not support appending)
            pass
    try:
        conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectorstore.as_retriever(), 
        memory=memory, 
        return_source_documents=True, 
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        )
    except Exception as e:
        st.error(f"Could not build the conversation chain. Please ensure your API key is valid: {e}")
        st.session_state.invalid_api_key = True
        return None
        return None
    return conversation_chain


# Local fallback removed — keeping the app simple: typed Google API key required for the conversation chain.

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
                st.session_state.vectorstore = vectorstore
                st.session_state.pdf_signature = compute_pdf_signature(pdf_docs)
                if api_key:
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore,
                        api_key,
                        language,
                        existing_messages=st.session_state.get('messages', []),
                    )
                else:
                    st.session_state.conversation = None
                status.update(label="✅ Ready!", state="complete", expanded=False)

elif process_button and not api_key: st.error("❌ Please enter your Google API Key.")

# Auto process when files are uploaded or changed
if pdf_docs:
    _ = process_documents_if_needed(pdf_docs, api_key, language)

# If vectorstore exists and conversation isn't set because the API key was missing earlier,
# create the chain now if the user entered an API key
if st.session_state.get('vectorstore') is not None and st.session_state.get('conversation') is None and api_key:
    try:
        st.session_state.conversation = get_conversation_chain(
            st.session_state.vectorstore,
            api_key,
            language,
            existing_messages=st.session_state.get('messages', []),
        )
        st.success("Conversation chain created with provided API key.")
    except Exception as e:
        st.error(f"Failed to create conversation chain: {e}")

# If language changed, and we have a vectorstore, re-create the conversation chain in the newly selected language.
if st.session_state.get('language_changed'):
    try:
        if st.session_state.get('vectorstore') is not None:
            if api_key:
                st.session_state.conversation = get_conversation_chain(
                    st.session_state.vectorstore,
                    api_key,
                    st.session_state.language_selected,
                    existing_messages=st.session_state.get('messages', []),
                )
                st.success(f"Conversation updated for language {st.session_state.language_selected}.")
            else:
                st.info(f"Vectorstore exists. Please add your API key to create a conversation chain for {st.session_state.language_selected}.")
        # clear the flag
        st.session_state.language_changed = False
    except Exception as e:
        st.error(f"Failed to update conversation for new language: {e}")

st.caption(f"Please ask your question in {st.session_state.get('language_selected', language)}.")

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
                    if st.session_state.conversation is None:
                        st.error("Conversation chain is not initialized. Check your API key and re-process the document.")
                        raise RuntimeError("Conversation not initialized")
                    response = st.session_state.conversation({'question': f"{user_question} (Answer in {language})"})
                    bot_response = response['answer']
                    st.markdown(bot_response)
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                except Exception as e:
                    err_str = str(e)
                    # Detect common invalid API key messages and show helpful guidance
                    if "API key not valid" in err_str or "API_KEY_INVALID" in err_str:
                        st.session_state.invalid_api_key = True
                        st.error("The Google API key appears to be invalid. Please update your key in the sidebar or set a valid GOOGLE_API_KEY in .env.")
                    else:
                        st.error(f"Error: {e}")
