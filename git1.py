import os
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import requests
import redis
import uuid
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import tempfile
from pathlib import Path
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import logging
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from io import BytesIO

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Initialize Qdrant client (vector database)
qdrant_client = QdrantClient("localhost", port=6333)

# Initialize sentence transformer model for embeddings
model = SentenceTransformer('all-mpnet-base-v2')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Groq API details
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Cloudflare R2 client setup
r2_client = boto3.client('s3',
    endpoint_url=os.getenv('R2_ENDPOINT_URL'),
    aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
    config=Config(signature_version='s3v4'),
    region_name='auto'
)

# R2 bucket name
R2_BUCKET_NAME = os.getenv('R2_BUCKET_NAME')

# Create Qdrant collection if it doesn't exist
qdrant_client.recreate_collection(
    collection_name="documents",
    vectors_config=models.VectorParams(size=model.get_sentence_embedding_dimension(), distance=models.Distance.COSINE),
)

class Query(BaseModel):
    text: str

def list_r2_files():
    """List all files in the R2 bucket."""
    try:
        response = r2_client.list_objects_v2(Bucket=R2_BUCKET_NAME)
        return [obj['Key'] for obj in response.get('Contents', [])]
    except ClientError as e:
        logger.error(f"Error listing R2 files: {e}")
        return []

def download_file_from_r2(file_key):
    """Download a file from R2 and return its content."""
    try:
        response = r2_client.get_object(Bucket=R2_BUCKET_NAME, Key=file_key)
        return BytesIO(response['Body'].read())
    except ClientError as e:
        logger.error(f"Error downloading file {file_key} from R2: {e}")
        return None

def compute_r2_file_hash(file_key):
    """Compute MD5 hash of a file stored in R2."""
    file_content = download_file_from_r2(file_key)
    if file_content:
        return hashlib.md5(file_content.getvalue()).hexdigest()
    return None

def process_r2_document(file_key):
    """Process a document stored in R2."""
    file_content = download_file_from_r2(file_key)
    if file_content:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file_content.getvalue())
            temp_file_path = temp_file.name

        loader = UnstructuredFileLoader(temp_file_path)
        documents = loader.load()
        
        os.unlink(temp_file_path)  # Clean up the temporary file

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        return [chunk.page_content for chunk in chunks]
    return []

def add_to_vector_db(chunks: List[str], file_path: str):
    file_id = str(uuid.uuid4())
    for chunk in chunks:
        embedding = model.encode(chunk)
        qdrant_client.upsert(
            collection_name="documents",
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={"text": chunk, "file_path": file_path, "file_id": file_id}
                )
            ]
        )
    return file_id

def process_existing_documents():
    processed_files = redis_client.get("processed_files")
    processed_files = json.loads(processed_files) if processed_files else {}

    r2_files = list_r2_files()
    files_to_process = []

    for file_key in r2_files:
        file_hash = compute_r2_file_hash(file_key)
        if file_key not in processed_files or processed_files[file_key] != file_hash:
            files_to_process.append((file_key, file_hash))

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(lambda x: process_single_r2_file(*x), files_to_process))

    for file_key, file_hash, file_id in results:
        if file_hash and file_id:
            processed_files[file_key] = file_hash

    redis_client.set("processed_files", json.dumps(processed_files))
    logger.info(f"Processed {len(files_to_process)} new or updated files from R2")

def process_single_r2_file(file_key, file_hash):
    try:
        chunks = process_r2_document(file_key)
        file_id = add_to_vector_db(chunks, file_key)
        return file_key, file_hash, file_id
    except Exception as e:
        logger.error(f"Error processing R2 file {file_key}: {str(e)}")
        return file_key, None, None

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        r2_client.put_object(Bucket=R2_BUCKET_NAME, Key=file.filename, Body=file_content)
        
        file_hash = hashlib.md5(file_content).hexdigest()
        chunks = process_r2_document(file.filename)
        file_id = add_to_vector_db(chunks, file.filename)

        processed_files = redis_client.get("processed_files")
        processed_files = json.loads(processed_files) if processed_files else {}
        processed_files[file.filename] = file_hash
        redis_client.set("processed_files", json.dumps(processed_files))

        return {"message": "Document uploaded to R2, processed, and added to the database", "file_id": file_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_existing_documents")
async def process_existing_docs(background_tasks: BackgroundTasks):
    background_tasks.add_task(process_existing_documents)
    return {"message": "Processing of existing documents has been initiated in the background"}

def retrieve_relevant_chunks(query: str, k: int = 3) -> List[str]:
    query_embedding = model.encode(query)
    
    cache_key = f"query:{query}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return json.loads(cached_result)
    
    search_result = qdrant_client.search(
        collection_name="documents",
        query_vector=query_embedding.tolist(),
        limit=k
    )
    
    relevant_chunks = [hit.payload['text'] for hit in search_result]
    
    redis_client.setex(cache_key, 3600, json.dumps(relevant_chunks))
    
    return relevant_chunks

def query_llama(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000
    }
    
    try:
        logger.info(f"Sending request to Groq API with prompt: {prompt[:50]}...")
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        response.raise_for_status()
        
        logger.info(f"Received response from Groq API. Status code: {response.status_code}")
        response_json: Dict[str, Any] = response.json()
        
        if 'choices' not in response_json or not response_json['choices']:
            logger.error(f"Unexpected response structure. Full response: {response_json}")
            return "Error: Unexpected response structure from Groq API"
        
        content = response_json['choices'][0]['message']['content']
        logger.info(f"Successfully extracted content from Groq API response")
        return content
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error occurred while querying Groq API: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response content: {e.response.text}")
        return f"Error querying Groq API: {str(e)}"

@app.post("/query")
async def query(query: Query):
    try:
        relevant_chunks = retrieve_relevant_chunks(query.text)
        prompt = f"Based on the following information:\n\n{' '.join(relevant_chunks)}\n\nAnswer the following question: {query.text}"
        response = query_llama(prompt)
        return {"response": response}
    except Exception as e:
        logger.exception("An error occurred while processing the query")
        raise HTTPException(status_code=500, detail=str(e))

def add_sample_data():
    sample_texts = [
        "The principle of conservation of energy states that energy cannot be created or destroyed, only converted from one form to another.",
        "In physics, work is the energy transferred to or from an object via the application of force along a displacement.",
        "Newton's first law of motion states that an object will remain at rest or in uniform motion in a straight line unless acted upon by an external force."
    ]
    for i, text in enumerate(sample_texts):
        file_key = f"sample_document_{i}.txt"
        r2_client.put_object(Bucket=R2_BUCKET_NAME, Key=file_key, Body=text.encode('utf-8'))
        chunks = [text]
        file_id = add_to_vector_db(chunks, file_key)
        logger.info(f"Added sample document with ID: {file_id}")

@app.on_event("startup")
async def startup_event():
    add_sample_data()

# Set up scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(
    process_existing_documents,
    trigger=IntervalTrigger(hours=24),
    id='process_documents_job',
    name='Process existing documents daily',
    replace_existing=True)
scheduler.start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
