import streamlit as st
import os
import glob
import torch

# Fix lỗi torch trên Windows
try:
    if hasattr(torch, 'classes'):
        torch.classes.__path__ = []
except:
    pass

from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="Legal Chatbot", layout="wide")
st.title("Trợ lý Luật sư AI")

# Sidebar
k_retrieval = st.sidebar.slider("Số lượng tài liệu tham khảo", 1, 10, 2)

# Path
DB_PATH = "chroma_db"
MODEL_DIR = "models"

@st.cache_resource
def load_resources():
    """Hàm load model và database, dùng cache để không load lại khi reload trang"""

    # 1. Tìm file model .gguf
    model_files = glob.glob(os.path.join(MODEL_DIR, "*.gguf"))
    if not model_files:
        return None, None
    
    model_path = model_files[0]
    print(f"Loading model from: {model_path}")

    # 2. Load Embedding Model bằng CPU
    embed = HuggingFaceEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder",
        model_kwargs={'device': 'cpu'}
    )

    # 3. Load LLM
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=8192,          # Context window
        n_batch=512,         # Batch size
        temperature=0.1,     # Giảm độ sáng tạo để bot trả lời đúng luật
        max_tokens=1024,     # Độ dài câu trả lời tối đa
        n_gpu_layers=0,      # Chạy CPU
        verbose=False
    )

    # 4. Load Vector Database
    db = None
    if os.path.exists(DB_PATH):
        db = Chroma(persist_directory=DB_PATH, embedding_function=embed)

    return llm, db

# Khởi tạo model
llm, db = load_resources()
if not llm:
    st.error("Lỗi: Không tìm thấy file .gguf trong thư mục models/. Vui lòng tải model về!")
    st.stop()

# Chatbot
if "history" not in st.session_state:
    st.session_state.history = [{"role": "assistant", "content": "Chào bạn, tôi là AI trợ lý pháp luật. Tôi có thể giúp gì được cho bạn nhỉ?"}]

# Hiển thị lịch sử chat
for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["content"])

# Xử lý khi người dùng nhập câu hỏi
if prompt := st.chat_input("Nhập câu hỏi..."):
    # Lưu và hiện câu hỏi của user
    st.session_state.history.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý..."):
            # Retrieve
            context = ""
            refs = []
            if db:
                docs = db.similarity_search(prompt, k=k_retrieval)
                context = "\n\n".join([d.page_content for d in docs])
                refs = docs
            
            # Prompt Engineering
            template = """### Instruction:
            Bạn là trợ lý luật sư. Trả lời dựa trên ngữ cảnh được cung cấp.
            Format câu trả lời: Căn cứ theo -> Nội dung -> Kết luận.
            
            Ngữ cảnh:
            {context}

            ### Input:
            {question}

            ### Response:"""
            
            p_template = PromptTemplate(template=template, input_variables=["context", "question"])
            final_prompt = p_template.format(context=context, question=prompt)
            
            # Generate
            response = llm.invoke(final_prompt)
            
            # References
            if refs:
                response += "\n\n**Nguồn:**\n" + "\n".join([f"- {d.metadata.get('source', 'Unknown')}" for d in refs])
            
            # Hiển thị và lưu câu trả lời
            st.write(response)
            st.session_state.history.append({"role": "assistant", "content": response})
