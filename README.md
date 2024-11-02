# Multilingual AI Chatbot with Dynamic Knowledge Retrieval for Enhanced Customer Experience

## Overview
This project is a **Multilingual AI Chatbot** designed to enhance customer experience by answering a wide range of queries across multiple languages. Built to handle diverse customer needs, the chatbot retrieves information dynamically from various sources, including FAQ databases, product manuals, and policy documents. Users can interact with the bot by uploading text or images, asking questions in any language, and receiving responses in their preferred language.

## Objective
To provide a **personalized and engaging customer support experience** by leveraging multilingual support and dynamic knowledge retrieval.

## Key Features
- **Multilingual Query and Response**: Users can ask questions in any language and receive responses in their preferred language, enhancing accessibility for global customers.
- **Dynamic Knowledge Retrieval**: The chatbot uses Retrieval-Augmented Generation (RAG) techniques to pull relevant information from diverse data sources in real-time, ensuring responses are accurate and up-to-date.
- **Text and Image Support**: Users can upload text or images with embedded text. The chatbot uses Optical Character Recognition (OCR) to extract text from images, enabling seamless information retrieval from various formats.

## Components
1. **Multilingual NLP and RAG**
   - Uses language-agnostic embeddings to facilitate cross-language data retrieval.
   - Integrates with LangChain to query multilingual documents effectively, allowing seamless conversations regardless of language differences.

2. **Real-Time Knowledge Base Updates**
   - The chatbot is continuously updated, pulling data from live databases and previous interactions to keep its knowledge base accurate and relevant.

## Technology Stack
- **Cohere Multilingual Model**: For embedding and cross-language data retrieval.
- **LangChain**: Enables RAG capabilities, allowing efficient querying across documents in multiple languages.
- **OCR Integration**: Extracts text from uploaded images for query and response processing.
- **Flask/HTML**: Provides a simple, user-friendly interface for interaction.


