Vietnam Legal AI Chatbot

A retrieval-augmented generation (RAG) application for querying and providing consultation on Vietnamese legal documents, powered by the Qwen2.5 Large Language Model (LLM). This system is designed to run offline on personal computers, ensuring data privacy and accessibility.

Overview

This project implements an AI assistant capable of answering legal inquiries by referencing actual legal texts. By leveraging RAG technology, the system minimizes hallucinations and provides accurate, citation-backed responses.

Key Features:

Offline Deployment: Runs entirely on local hardware (CPU or GPU).

Accurate Retrieval: Uses hybrid search (Keyword + Vector) to find relevant legal articles.

Citation Support: Every answer includes specific references to the legal source (Article, Clause, Law).

Hardware Optimization: Supports GGUF model format for efficient execution on consumer-grade hardware.

Installation

1. System Requirements

Operating System: Windows 10/11, Linux, or macOS.

Python: Version 3.10 or higher.

RAM: Minimum 8GB (16GB recommended).

GPU (Optional): NVIDIA RTX 3050 or better for accelerated performance (requires CUDA).

2. Install Dependencies

It is recommended to use a virtual environment or Anaconda.

Standard Installation (CPU/GPU auto-detection):

pip install -r requirements.txt


Specific Installation for GPU Acceleration (NVIDIA):

If you have an NVIDIA GPU and CUDA Toolkit installed, run the following commands to enable GPU support for llama-cpp-python:

(For Windows PowerShell)

$env:CMAKE_ARGS = "-DGGML_CUDA=on"
pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade


(For Linux/Mac)

CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade


Model and Data Setup

Due to file size limitations on GitHub, the AI Model (~5GB) and Vector Database are not included in the repository. Please follow these steps to download them manually.

Step 1: Download AI Model (Qwen2.5-7B)

Run the provided script to automatically download the GGUF model from Hugging Face:

python download_7b_model.py


This script will download the qwen2.5-7b-instruct-q4_k_m.gguf file and save it to the models/ directory.
Note: The download process may take 10-15 minutes depending on your internet connection.

Step 2: Prepare Legal Database (ChromaDB)

You need a pre-processed vector database to run the search engine.

Create a folder named chroma_db in the root directory.

If you have your own PDF legal documents, place them in a dataset/ folder and run the ingestion script (if available).

Recommended: Use the pre-built chroma_db folder provided by the project maintainer (external download link) and extract it into the project root.


Running the Application

Open your terminal in the project directory and execute:

streamlit run app_ui.py


The application will launch automatically in your default web browser at http://localhost:8501.

Configuration

You can adjust performance settings in app_ui.py to match your hardware capabilities:

n_ctx: Context window size (Default: 8192). Reduce to 4096 if you encounter Out-Of-Memory (OOM) errors.

n_gpu_layers: Number of model layers offloaded to the GPU.

Set to 0 for CPU-only mode (Slower).

Set to 20-30 for GPU acceleration (Faster, depends on VRAM).

Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the issues page if you want to contribute.

License

This project is distributed under the MIT License. See the LICENSE file for more information.
