# ReAI
## Overview
The ReAI System is a web application built with Flask that integrates a Retrieval-Augmented Generation (RAG) model for answering questions. It uses meta Llama3 and cohere embed-english hosted on AWS Bedrock to process and respond to user queries. This system is designed to handle both structured and unstructured data efficiently.

## Features
1. Query Classification: Classifies queries into SQL-related or vector store-related.
2. SQL Query Generation: Generates SQL queries based on user prompts and retrieves results from a PostgreSQL database.
3. Natural Language Answering: Converts SQL results into natural language responses using an LLM.
4. Vector Store Querying: Retrieves relevant documents from a vector store and generates answers based on these documents.
5. File Processing: Processes files from Google Drive, generates embeddings, and loads them into ChromaDB.

## Setup
Ensure python version 3 is installed on your machine.

First create and activate virtual environment
```
python3 -m venv venv
source venv/bin/activate
```
clone github repository and install dependencies
```
git clone https://github.com/Aniket0898/ReAI.git
cd ReAI
pip install -r requirements.txt
```
## configure aws-cli
You should have created acccess keys an AWS IAM user with permission for bedrock access. Execute command for configuring keys.
```
aws configure
```
## Update .env file
```
GDRIVE_API_KEY="your_google_apis_key
AWS_ACCESS_KEY_ID="your_aws_access_key"
AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
REGION_NAME="your_aws_region"
CHROMA_HOST_IP="your_chroma_host_ip"
CHROMA_PORT="your_chroma_host_port"
POSTGRES_USER="your_postgres_db_username"
POSTGRES_PASSWORD="your_postgres_db_password"
POSTGRES_HOST_IP="your_postgres_host_ip"
POSTGRES_DB="your_postgres_db_name"
```
## Run the Application
Start the Flask application:
```
python3 main.py
```
## Endpoints
- /query
- Method: POST

- Description: Receives a user question, classifies the prompt, and routes it to either SQL database querying or vector store querying.

Request Body:
```
{
    "question": "What is the average price of houses with 3 bedrooms?"
}
```
- /process_files
- Method: POST

- Description: Downloads files from Google Drive Folder which has public access, processes them to extract text, generates embeddings, and loads them into ChromaDB.

Request Body:
```
{
    "folder_url": "https://drive.google.com/drive/folders/folder-id"
}
```
