# ReAI
Overview
The ReAI System is a web application built with Flask that integrates a Retrieval-Augmented Generation (RAG) model for answering questions. It uses various technologies, including AWS Bedrock to process and respond to user queries. This system is designed to handle both structured and unstructured data efficiently.

Features
Query Classification: Classifies queries into SQL-related or vector store-related.
SQL Query Generation: Generates SQL queries based on user prompts and retrieves results from a PostgreSQL database.
Natural Language Answering: Converts SQL results into natural language responses using an LLM.
Vector Store Querying: Retrieves relevant documents from a vector store and generates answers based on these documents.
File Processing: Processes files from Google Drive, generates embeddings, and loads them into ChromaDB.
