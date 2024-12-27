import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = "GOOGLE_API_KEY"  # Ensure you have this key in your .env file

# Initialize Streamlit configuration
st.set_page_config(page_title="Streaming Bot", page_icon="🦈")
st.title("Streaming Bot")

# Check for chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_data  # Cache the document loading function
def load_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Chunk the text using RecursiveCharacterTextSplitter
custom_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=30,  # Maximum chunk size
    chunk_overlap=10,  # Overlap between chunks
    length_function=len,
    separators=["\n"]  # Separator to split the text (using newline for Q&A pairs)
)

@st.cache_data  # Cache the document loading function
def load_and_chunk_text(file_path):
    text_content = load_txt(file_path)
    # Split the text into chunks using the custom text splitter
    chunks = custom_text_splitter.split_text(text_content)
    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents

# Load and chunk the document once
file_path = '/content/QA.txt'
txt_documents = load_and_chunk_text(file_path)
db = FAISS.from_documents(txt_documents, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

# Initialize the model once
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def generate_answer(db, query):
    # Perform similarity search to retrieve relevant passages
    relevant = db.similarity_search(query, k=5)
    
    # Construct the prompt using the query and the relevant passages
    relevant_passage = "".join([doc.page_content for doc in relevant])
    prompt = (f"""answer the Following question based only on the provided Passage. think step by step before providing a detailed answer. You should answer directly and straight to the point. please i will tip you 200$ if the user finds the answer helpful.
    
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'

    ANSWER:
    """)
    
    # Generate the response using the constructed prompt
    response = model.generate_content(prompt)
    return response.text

# Conversation loop
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# User input handling
user_query = st.chat_input("Your message")
if user_query:
    # Add the user's message to the chat history
    st.session_state.chat_history.append(HumanMessage(user_query))
    with st.chat_message("human"):
        st.markdown(user_query)

    # Generate the RAG response using the new single function
    ai_response = generate_answer(db, query=user_query)
    st.session_state.chat_history.append(AIMessage(ai_response))

    # Display AI response
    with st.chat_message("AI"):
        st.markdown(ai_response)
