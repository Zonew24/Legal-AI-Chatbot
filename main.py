import streamlit as st
import os
import glob
import torch


# Fix lỗi xung đột torch và streamlit trên Windows
try:
    if hasattr(torch, 'classes'):
        torch.classes.__path__ = []
except:
    pass

from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

# Config
st.set_page_config(page_title="Legal Chatbot", layout="wide")
st.title("Legal AI Chatbot")

# Sidebar chỉnh tham số
with st.sidebar:
    st.header("Cấu hình")
    k_retrieval = st.slider("Số docs tham khảo (K)", 1, 10, 2, help="Ít docs thì chạy nhanh hơn")

# Model
DB_PATH = "chroma_db"
MODEL_DIR = "models"

def get_model_path():
    print(f"Scanning models in: {os.path.abspath(MODEL_DIR)}")
    # Tìm tất cả file .gguf
    all_ggufs = glob.glob(os.path.join(MODEL_DIR, "*.gguf"))
    
    if not all_ggufs:
        return None
        
    # Ưu tiên tìm bản 7B
    for path in all_ggufs:
        if "7b" in path.lower():
            return path
    return all_ggufs[0]

# Cache resource để không phải load lại model mỗi lần F5
@st.cache_resource
def load_engine():
    model_path = get_model_path()
    
    # In ra đường dẫn
    if model_path:
        print(f"--> Found Model: {model_path}")
    else:
        print("--> No model found!")
        return None, None

    # Embedding model (Chạy CPU ok)
    embed = HuggingFaceEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder",
        model_kwargs={'device': 'cpu'}
    )
    
    # Load LLM (LlamaCpp)
    # n_gpu_layers=0 nghĩa là chạy full CPU. Nếu có Card rời thì tăng lên 20-30.
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=8192,
        n_batch=512,
        temperature=0.1,
        max_tokens=1024,
        n_gpu_layers=0,  # Chạy full CPU.
        verbose=False
    )
    
    # Load Vector DB
    db = None
    if os.path.exists(DB_PATH):
        db = Chroma(persist_directory=DB_PATH, embedding_function=embed)
        
    return llm, db

llm, db = load_engine()

# Chat UI
if "history" not in st.session_state:
    st.session_state.history = [{"role": "assistant", "content": "Chào bạn, tôi là AI tư vấn luật tại Việt Nam. Tôi có thể giúp gì cho bạn hôm nay?"}]

# Render tin nhắn cũ
for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["content"])

# Xử lý tin nhắn mới
if user_input := st.chat_input("Nhập câu hỏi..."):
    # Hiện câu hỏi user
    st.session_state.history.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            
            # Retrieve
            context_text = ""
            refs = []
            if db:
                docs = db.similarity_search(user_input, k=k_retrieval)
                context_text = "\n\n".join([d.page_content for d in docs])
                refs = docs
            
            #Prompting
            template = """<|im_start|>system
            Bạn là trợ lý luật sư. Trả lời dựa trên ngữ cảnh được cung cấp.
            Format: Căn cứ theo -> Nội dung -> Kết luận.
            
            Ngữ cảnh:
            {context}<|im_end|>
            <|im_start|>user
            {question}<|im_end|>
            <|im_start|>assistant"""
            
            prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])
            final_prompt = prompt_template.format(context=context_text, question=user_input)
            
            # Generate
            response = llm.invoke(final_prompt)
            
            # Source
            if refs:
                response += "\n\n**Nguồn tham khảo:**\n" + "\n".join([f"- {d.metadata.get('source', 'Unknown')}" for d in refs])
            
            st.write(response)
            st.session_state.history.append({"role": "assistant", "content": response})