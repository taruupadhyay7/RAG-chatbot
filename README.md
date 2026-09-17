# RAG-chatbot

A Retrieval-Augmented Generation (RAG) based chatbot for document Q&A, built as a company assignment.

## What It Does
Answers user questions by retrieving relevant context from documents and generating grounded responses.

## Tech Stack
- **LLaMA 2** — response generation
- **Streamlit** — interactive chat interface

## How It Works
1. User uploads/queries a document
2. Relevant context is retrieved
3. LLaMA 2 generates an answer grounded in that context, served through a Streamlit chat UI
