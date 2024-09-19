# import libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_aws import BedrockLLM, BedrockEmbeddings
from sqlalchemy import create_engine, text
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import chromadb
from file_handler import FileHandler
import os
import re

app = Flask(__name__)

load_dotenv()

GDRIVE_API_KEY = os.getenv('GDRIVE_API_KEY')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
REGION_NAME = os.getenv('REGION_NAME')
CHROMA_HOST_IP = os.getenv('CHROMA_HOST_IP')
CHROMA_PORT = os.getenv('CHROMA_PORT')
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_HOST_IP = os.getenv('POSTGRES_HOST_IP')
POSTGRES_DB = os.getenv('POSTGRES_DB')

print(POSTGRES_DB)
# Define AWS credentials (replace with your actual credentials or secure storage)
os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY

# Initialize embedding model with boto3 session
embed_model = BedrockEmbeddings(
    credentials_profile_name=None, 
    region_name="us-east-1",
    model_id="cohere.embed-english-v3"
)

llm = BedrockLLM(
    credentials_profile_name=None, 
    region_name="us-east-1",
    model_id="meta.llama3-8b-instruct-v1:0",
    model_kwargs={"temperature": 0.5}
)

# Initialize remote ChromaDB
chromadb_client = chromadb.HttpClient(host=CHROMA_HOST_IP, port=CHROMA_PORT)
chroma_collection = chromadb_client.get_or_create_collection('qna-embeddings')

# Initialize PostgreSQL client
postgres_engine = create_engine(f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST_IP}/{POSTGRES_DB}')

# Function to classify prompt and route accordingly
def classify_and_route_prompt(prompt):
    # Define keywords for SQL database routing
    sql_keywords = ["price", "area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement",
                    "hotwaterheating", "airconditioning", "parking", "prefarea", "furnishingstatus"]

    # Check if prompt contains any SQL-related keywords
    if any(keyword in prompt.lower() for keyword in sql_keywords):
        # Route to SQL database query
        return query_sql_database(prompt), None
    else:
        # Route to vector store
        return vector_store(prompt)

# Function to query the SQL database
def query_sql_database(prompt):
    # Generate SQL query using LLM
    sql_query = generate_sql_query_from_prompt(prompt)

    # Ensure sql_query is a string
    if not isinstance(sql_query, str):
        raise ValueError("The SQL query must be a string.")

    # Execute SQL query
    with postgres_engine.connect() as conn:
        result = conn.execute(text(sql_query))
        response = result.fetchall()

    # Convert the result to a string for LLM input
    result_str = "\n".join(str(row) for row in response)

    # Generate natural language answer using LLM
    answer = generate_sql_answer(prompt, sql_query, result_str)
    return answer

def generate_sql_query_from_prompt(prompt):
    # Call LLM to generate SQL query based on the prompt
    
    system_message = """
        You are a data analyst.
        You are interacting with a user who is asking you questions about the company's database.
        Based on the database table real_estate schema below, write a SQL query that would answer the user's question.

        table name : real_estate
        Here is the table schema:
        <schema> 
        id: SERIAL (Primary Key)
        price: NUMERIC
        area: INTEGER
        bedrooms: INTEGER
        bathrooms: INTEGER
        stories: INTEGER
        mainroad: BOOLEAN (values: yes or no)
        guestroom: BOOLEAN (values: yes or no)
        basement: BOOLEAN (values: yes or no)
        hotwaterheating: BOOLEAN (values: yes or no)
        airconditioning: BOOLEAN (values: yes or no)
        parking: INTEGER
        prefarea: BOOLEAN (values: yes or no)
        furnishingstatus: VARCHAR(20)
        </schema>
        Write ONLY THE SQL QUERY and nothing else. 
        Do not wrap the SQL query in any other text, not even backticks.
        For example:
        Question: Name 10 customers
        SQL Query: SELECT Name FROM Customers LIMIT 10;
        Your turn:
        Question: {question}
        SQL Query:
        """
    human_message = [{"type": "text", "text": prompt}] 

    llm_template = [
        ("system", system_message),
        ("human", human_message),
    ] 

    # Invoke LLM to generate SQL query
    sql_query = llm.invoke(input=llm_template)
    
    # Extract the SQL query from the LLM response using a regular expression
    match = re.search(r"SELECT.*?;", sql_query, re.DOTALL) 
    if match:
        sql_query = match.group(0).strip()
    else:
        sql_query = ""  # Handle cases where no SQL query is found
    # Ensure sql_query is a string and handle LLM response
    if not isinstance(sql_query, str):
        raise ValueError("The generated SQL query must be a string.")
    
    return sql_query.strip()

# Function to generate natural language answer from SQL result
def generate_sql_answer(question, query, result):
    # Define the prompt template for generating the natural language answer
    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

        Question: {question}
        SQL Query: {query}
        SQL Result: {result}
        Answer: """
    )
    
    # Prepare the prompt
    formatted_prompt = answer_prompt.format(question=question, query=query, result=result)

    # Generate the answer with the LLM
    response = llm.invoke(input=formatted_prompt)

    return response

# Function to query the vector store
def vector_store(prompt):
    # Generate embeddings for the user question
    question_embedding = embed_model.embed_query(prompt)

    # Query the ChromaDB collection
    results = chroma_collection.query(
        query_embeddings=[question_embedding], 
        n_results=5
    )
    documents = results.get('documents', [])
    metadatas = results.get('metadatas', [])

    # Prepare context from the retrieved documents
    retrieved_docs = [doc[0] for doc in documents]
    context = "\n\n".join(retrieved_docs)

    # Define the prompt template
    prompt_template = """Use the following pieces of context to answer the question at the end.

    {context}

    Question: {question}
    Answer: """
    
    # Instantiate PromptTemplate
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Prepare the prompt
    formatted_prompt = PROMPT.format(context=context, question=prompt)

    # Generate the answer with the LLM
    response = llm.invoke(input=formatted_prompt)

    return response, metadatas

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_question = data.get('question', '')

    # Classify and route the user question
    response, metadatas = classify_and_route_prompt(user_question)

    return jsonify({'response': response, 'metadata': metadatas})

# Route to download, load, and generate embeddings and loading to chromadb for Google Drive files
@app.route('/process_files', methods=['POST'])
def process_files():
    folder_url = request.json.get('folder_url')
    api_key = os.getenv('GDRIVE_API_KEY')
    file_handler = FileHandler(api_key)

    # Process files using the FileHandler (downloads, extracts text, returns documents)
    documents = file_handler.process_files(folder_url)

    # Create text documents with required metadata from documents list
    text_documents = [{'id': doc.id_, 'metadata': doc.metadata, 'text': doc.text} for doc in documents]

    # Generating embeddings for text documents with embedding model and loading to ChromaDB
    embeddings = embed_model.embed_documents([doc['text'] for doc in text_documents])
    for doc, embedding in zip(text_documents, embeddings):
        chroma_collection.add_document(doc['id'], embedding, doc['metadata'])

    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
