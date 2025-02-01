import os
import json
from pinecone import Pinecone, ServerlessSpec
import tiktoken
import numpy as np
import openai
from tqdm import tqdm
from dotenv import load_dotenv
import re
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
INDEX_NAME = os.getenv('INDEX_NAME', 'upwork-v1')  # Single index name
EMBED_MODEL = os.getenv('EMBED_MODEL', 'text-embedding-ada-002')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '700'))  # Tokens per chunk
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))  # Token overlap between chunks
BATCH_SIZE = 100  # Pinecone's recommended batch size

# Initialize OpenAI client
client = openai.OpenAI()

def get_embeddings_batch(texts):
    """Generate embeddings for a batch of texts using OpenAI"""
    try:
        # Add retry logic for API calls
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    input=texts,
                    model=EMBED_MODEL
                )
                return [embedding.embedding for embedding in response.data]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

############################################
# Token-based chunking
############################################
def tokenize_text(text, encoder):
    return encoder.encode(text, disallowed_special=())

def detokenize_text(tokens, encoder):
    return encoder.decode(tokens)

def chunk_text(text, chunk_size, encoder):
    """Split text into roughly chunk_size tokens per chunk."""
    tokens = tokenize_text(text, encoder)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = detokenize_text(tokens[start:end], encoder)
        chunks.append(chunk)
        start = end
    return chunks

############################################
# Pinecone initialization
############################################
def init_pinecone(force_recreate=True):
    """Initialize Pinecone with optional index recreation."""
    print("Initializing Pinecone...")
    pc = Pinecone(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    
    # Check if index exists
    existing_indexes = pc.list_indexes().names()
    
    if INDEX_NAME in existing_indexes:
        if force_recreate:
            print(f"Deleting existing index '{INDEX_NAME}'...")
            pc.delete_index(INDEX_NAME)
            print("Waiting for deletion to complete...")
            time.sleep(20)  # Give more time for deletion
            print("Creating new index...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            print("Index created successfully.")
            time.sleep(20)  # Wait longer for index to be fully ready
        else:
            print(f"Using existing index '{INDEX_NAME}'")
    else:
        print(f"Creating new index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print("Index created successfully.")
        time.sleep(20)  # Wait for index to be ready
    
    return pc.Index(INDEX_NAME)

############################################
# Main script
############################################
def clean_for_embedding(text):
    """Clean text before generating embeddings"""
    if not text or not isinstance(text, str):
        return ""
        
    # Remove any remaining non-human-readable content
    text = re.sub(r'[^\x20-\x7E\n\t]', '', text)  # Keep only printable ASCII
    
    # Remove any remaining technical artifacts
    text = re.sub(r'undefined|null|NaN|Infinity|-Infinity', '', text)
    text = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '', text)  # Remove UUIDs
    text = re.sub(r'[\{\[\(].*?[\}\]\)]', '', text)  # Remove bracketed content
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def chunk_text_with_overlap(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into chunks with overlap using tiktoken."""
    # Initialize encoder
    encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
    
    # Tokenize the text
    tokens = encoder.encode(text)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        # Define chunk end with overlap
        end = min(start + chunk_size, len(tokens))
        
        # Get the chunk and decode back to text
        chunk = encoder.decode(tokens[start:end])
        chunks.append(chunk)
        
        # Move start position, accounting for overlap
        start = end - overlap if end < len(tokens) else end
    
    return chunks

def generate_unique_doc_id(doc_type, identifier, timestamp=None):
    """Generate a unique document ID using type, identifier, and optional timestamp."""
    if timestamp is None:
        timestamp = int(time.time())
    # Create a hash of the identifier to keep ID length reasonable
    id_hash = hex(hash(identifier) & 0xFFFFFF)[2:]  # Take last 6 chars of hash
    return f"{doc_type}_{id_hash}_{timestamp}"

def process_jobs_history(jobs_data):
    """Process jobs with enhanced metadata and unique IDs."""
    docs = []
    for job in jobs_data["jobs"]:
        # Use the uid from the JSON data
        unique_id = job.get('uid')
        if not unique_id:
            # Fallback to generating one if uid doesn't exist
            start_date = job.get('date_started') or "unknown_date"
            unique_id = generate_unique_doc_id(
                "job",
                f"{job['title']}_{start_date}"
            )
        
        # Extract client feedback text cleanly
        client_feedback = job.get('job_feedback', {}).get('client_feedback_given', {})
        feedback_text = client_feedback.get('text', '') if isinstance(client_feedback, dict) else ''
        feedback_rating = client_feedback.get('rating', 0) if isinstance(client_feedback, dict) else 0

        # Focus on the relevant content
        proposal_content = f"""
        Job Type: {job.get('job_details', {}).get('job_type', 'N/A')}
        Experience Level: {job.get('job_details', {}).get('experience_level', 'N/A')}
        
        Original Job Description:
        {job.get('job_details', {}).get('job_description', 'N/A')}
        
        My Successful Proposal:
        {job.get('cover_letter', 'N/A')}
        
        Client Feedback:
        {feedback_text}
        """
        
        metadata = {
            "type": "job_history",
            "source": "jobs_history.json",
            "doc_id": unique_id,
            "title": str(job['title']),
            "date_started": str(job.get('date_started') or "N/A"),
            "date_ended": str(job.get('date_ended') or "N/A"),
            "total_earned": float(job.get('total_earned', 0)),
            "hourly_rate": float(job.get('hourly_rate', 0)),
            "hours_spent": float(job.get('time_spent_hours', 0)),
            "job_type": str(job.get('job_details', {}).get('job_type') or "N/A"),
            "experience_level": str(job.get('job_details', {}).get('experience_level') or "N/A"),
            "client_rating": float(feedback_rating),
            "client_location": str(job.get('client_info', {}).get('location') or "N/A"),
            "client_total_spent": float(job.get('client_info', {}).get('total_spent', 0)),
            "client_hires": int(job.get('client_info', {}).get('hires', 0)),
            "has_feedback": bool(feedback_text),
            "has_cover_letter": bool(job.get('cover_letter')),
            "processing_timestamp": int(time.time())
        }
        
        docs.append({
            "id": unique_id,
            "content": proposal_content,
            "metadata": metadata
        })
    return docs

def process_profile_data(profile_data):
    """Process profiles with enhanced metadata and unique IDs."""
    docs = []
    for profile in profile_data["profiles"]:
        # Create unique ID using specialization and current timestamp
        unique_id = generate_unique_doc_id(
            "profile",
            profile['specialization']
        )
        
        # Create comprehensive profile text
        profile_text = f"""
        Title: {profile['title']}
        Specialization: {profile['specialization']}
        Hourly Rate: ${profile['hourly_rate']}
        
        Overview: {profile['overview']}
        
        Key Achievements: {' '.join(profile.get('key_achievements', []))}
        
        Areas of Expertise: {' '.join(profile.get('areas_of_expertise', []))}
        
        Notable Clients: {' '.join(profile.get('notable_clients', []))}
        
        Approach: {profile['approach']}
        
        Technical Skills: {', '.join(profile.get('technical_expertise', {}).get('languages', []) + 
                                   profile.get('technical_expertise', {}).get('frameworks', []) +
                                   profile.get('technical_expertise', {}).get('tools', []))}
        """
        
        metadata = {
            "type": "profile",
            "source": "profile_data.json",
            "doc_id": unique_id,
            "title": profile['title'],
            "specialization": profile['specialization'],
            "hourly_rate": float(profile['hourly_rate']),
            "has_key_achievements": bool(profile.get('key_achievements')),
            "notable_clients_count": len(profile.get('notable_clients', [])),
            "skills_count": len(profile.get('technical_expertise', {}).get('languages', []) +
                              profile.get('technical_expertise', {}).get('frameworks', []) +
                              profile.get('technical_expertise', {}).get('tools', [])),
            "languages": profile.get('technical_expertise', {}).get('languages', []),
            "frameworks": profile.get('technical_expertise', {}).get('frameworks', []),
            "tools": profile.get('technical_expertise', {}).get('tools', []),
            "processing_timestamp": int(time.time())
        }
        
        docs.append({
            "id": unique_id,
            "content": profile_text,
            "metadata": metadata
        })
    return docs

def process_and_upload_documents(docs, index_name):
    """Process and upload documents to a specific index with batch processing"""
    global INDEX_NAME
    INDEX_NAME = index_name
    
    # Initialize Pinecone
    index = init_pinecone()
    
    # Process documents
    print(f"\nProcessing documents for {index_name}...")
    vectors_to_upsert = []
    total_chunks = 0
    current_batch_texts = []
    current_batch_metadata = []
    current_batch_ids = []
    
    for doc in tqdm(docs, desc="Processing documents"):
        # Clean the content
        cleaned_content = clean_for_embedding(doc["content"])
        
        if len(cleaned_content.strip()) < 20:
            print(f"\nSkipping document '{doc['metadata']['title']}': Too short after cleaning")
            continue
        
        # Split into chunks
        content_chunks = chunk_text_with_overlap(cleaned_content)
        
        for chunk_index, chunk in enumerate(content_chunks):
            # Prepare metadata
            metadata = {
                **doc['metadata'],
                "chunk_index": chunk_index,
                "total_chunks": len(content_chunks),
                "text": chunk
            }
            
            # Prepare document ID
            doc_id = f"{doc['id']}_chunk_{chunk_index}" if len(content_chunks) > 1 else doc['id']
            
            # Add to current batch
            current_batch_texts.append(chunk)
            current_batch_metadata.append(metadata)
            current_batch_ids.append(doc_id)
            
            # Process batch if size reached
            if len(current_batch_texts) >= BATCH_SIZE:
                embeddings = get_embeddings_batch(current_batch_texts)
                if embeddings:
                    vectors_to_upsert.extend([
                        (id, emb, meta) 
                        for id, emb, meta in zip(current_batch_ids, embeddings, current_batch_metadata)
                    ])
                    total_chunks += len(embeddings)
                    
                    # Upsert vectors
                    index.upsert(vectors=vectors_to_upsert)
                    vectors_to_upsert = []
                
                # Clear batches
                current_batch_texts = []
                current_batch_metadata = []
                current_batch_ids = []
    
    # Process any remaining items
    if current_batch_texts:
        embeddings = get_embeddings_batch(current_batch_texts)
        if embeddings:
            vectors_to_upsert.extend([
                (id, emb, meta)
                for id, emb, meta in zip(current_batch_ids, embeddings, current_batch_metadata)
            ])
            total_chunks += len(embeddings)
    
    # Upsert any remaining vectors
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)
    
    return total_chunks

def main():
    print("\n=== Starting Document Processing ===\n")
    
    # Load the JSON files
    print("Loading documents...")
    with open("jobs_history.json", "r", encoding="utf-8") as f:
        jobs_data = json.load(f)
    with open("profile_data.json", "r", encoding="utf-8") as f:
        profile_data = json.load(f)
    
    # Process all documents
    all_docs = []
    all_docs.extend(process_jobs_history(jobs_data))
    all_docs.extend(process_profile_data(profile_data))
    
    # Upload to single index with force_recreate=True
    total_chunks = process_and_upload_documents(all_docs, INDEX_NAME)

    print("\n=== Processing Complete ===")
    print(f"Total documents: {len(all_docs)}")
    print(f"Total chunks: {total_chunks}")

if __name__ == "__main__":
    main() 