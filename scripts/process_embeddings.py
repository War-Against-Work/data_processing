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
from typing import List

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

def _derive_domain_patterns(job_description: str) -> List[str]:
    """Simplified fallback pattern-based domain detection."""
    if not job_description:
        return ["general"]
        
    text = job_description.lower()
    domains = set()  # Using set to avoid duplicates
    
    # Simple keyword-based checks for major domains
    basic_domains = {
        "finance": ["finance", "bank", "invest", "trading", "accounting"],
        "healthcare": ["health", "medical", "clinic", "patient"],
        "technology": ["tech", "software", "data", "cloud", "ai"],
        "marketing": ["market", "brand", "content", "seo", "advertising"],
        "education": ["education", "learn", "teach", "training", "course"],
        "ecommerce": ["commerce", "shop", "retail", "marketplace"]
    }
    
    # Check for domain keywords
    for domain, keywords in basic_domains.items():
        if any(keyword in text for keyword in keywords):
            domains.add(domain)
    
    # Add development categories if detected
    if any(keyword in text for keyword in ["web", "website", "frontend", "backend"]):
        domains.add("web_development")
    if any(keyword in text for keyword in ["mobile", "ios", "android", "app"]):
        domains.add("mobile_development")
    
    return list(domains) if domains else ["general"]

async def _derive_domain_llm(job_description: str, client: openai.OpenAI) -> List[str]:
    """Use LLM to derive domains with sophisticated analysis."""
    try:
        messages = [
            {"role": "developer", "content": """You are a domain classification expert.
             Given a job description, identify the most relevant business and technical domains.
             Consider both explicit mentions and implicit requirements.
             Think about both the business context and technical requirements.
             
             Return a JSON object with a "domains" array containing relevant domain strings.
             Be specific but avoid over-categorization."""},
            {"role": "user", "content": f"Analyze this job description and identify its domains:\n{job_description}"}
        ]
        
        completion = client.chat.completions.create(
            model="o3-mini",
            messages=messages,
            reasoning_effort="medium",
            response_format={"type": "json_object"},
            max_completion_tokens=1000
        )
        
        try:
            domains = json.loads(completion.choices[0].message.content).get("domains", [])
            if not domains:  # If LLM returns empty list
                logger.warning("LLM returned empty domains list, using pattern fallback")
                return _derive_domain_patterns(job_description)
                
            logger.info(f"LLM detected domains: {domains}")
            return domains
            
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse LLM domain response: {e}, using pattern fallback")
            return _derive_domain_patterns(job_description)
            
    except Exception as e:
        logger.error(f"LLM domain analysis failed: {str(e)}, using pattern fallback")
        return _derive_domain_patterns(job_description)

async def _analyze_style_and_tone(text: str, client: openai.OpenAI) -> dict:
    """Dynamically analyze writing style, technical depth, and tone with chain-of-thought reasoning."""
    try:
        messages = [
            {"role": "developer", "content": """You are an expert proposal analyst.
             Analyze the provided text and think about its overall tone, technical depth, and voice style.
             Consider how the writing connects with the reader, demonstrates expertise, and conveys value.
             
             Return a JSON object with these keys:
             - overall_tone: Your assessment of the primary tone
             - technical_depth: Level of technical sophistication
             - voice_style: The writing approach and style used
             - raw_analysis: Your complete analysis and reasoning"""},
            {"role": "user", "content": f"Analyze this proposal text:\n{text}"}
        ]
        
        completion = client.chat.completions.create(
            model="o3-mini",
            messages=messages,
            reasoning_effort="medium",  # Using medium reasoning effort
            max_completion_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        try:
            analysis = json.loads(completion.choices[0].message.content)
            logger.info(f"Dynamic style analysis completed: {analysis}")
            return analysis
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse dynamic style analysis: {e}, returning default analysis")
            return _default_style_analysis()
            
    except Exception as e:
        logger.error(f"Dynamic style analysis failed: {str(e)}")
        return _default_style_analysis()

def _default_style_analysis() -> dict:
    """Default analysis in case LLM fails."""
    return {
        "overall_tone": "professional",
        "technical_depth": "intermediate",
        "voice_style": "direct",
        "raw_analysis": ""
    }

async def process_jobs_history(jobs_data):
    """Process jobs with enhanced metadata for learning and pattern recognition."""
    docs = []
    
    for job in jobs_data:
        # Parse status/outcome - use outcome as the primary status field
        raw_status = job.get('outcome', job.get('status', '')).lower()
        
        # Determine status type
        status_mapping = {
            'hired': 'hired',
            'job is closed': 'closed',
        }
        
        # Check for date format
        is_date = bool(re.match(r'[A-Za-z]+ \d{1,2}, \d{4}', raw_status))
        
        # Determine final status
        if is_date:
            status_type = 'date'
        else:
            status_type = next((v for k, v in status_mapping.items() if k in raw_status), 'other')
        
        # Extract client info from structured text
        client_info = job.get('client_info', {})
        
        # Parse numeric values from text
        def extract_number(text, pattern, convert_k=False):
            if not text:
                return None
            match = re.search(pattern, text)
            if not match:
                return None
            try:
                value = float(match.group(1))
                if convert_k and 'K' in text:
                    value *= 1000
                return value
            except (ValueError, TypeError):
                return None
        
        # Extract client metrics with improved parsing
        total_spent = extract_number(client_info.get('total_spent', ''), r'\$?([\d.]+)', convert_k=True)
        avg_hourly = extract_number(client_info.get('avg_hourly_rate', ''), r'\$?([\d.]+)')
        total_hours = extract_number(client_info.get('hours', ''), r'([\d.]+)')
        
        # Parse hire info with better error handling
        hire_info = client_info.get('hire_info', '')
        hire_rate = extract_number(hire_info, r'(\d+)%')
        has_open_jobs = 'open job' in hire_info.lower()
        
        # Parse hires info with null safety
        hires_info = client_info.get('hires_info', '') or ''
        hires_match = re.search(r'(\d+) hires?, (\d+) active', hires_info)
        total_hires = int(hires_match.group(1)) if hires_match else 0
        active_hires = int(hires_match.group(2)) if hires_match else 0
        
        # Parse rating and reviews more robustly
        rating = extract_number(client_info.get('rating', ''), r'([\d.]+)')
        reviews_match = re.search(r'([\d.]+) of (\d+) reviews', client_info.get('reviews', '') or '')
        review_score = float(reviews_match.group(1)) if reviews_match else None
        review_count = int(reviews_match.group(2)) if reviews_match else 0
        
        # Derive domains using LLM with pattern matching fallback
        try:
            domains = await _derive_domain_llm(job.get('job_description', ''), client)
        except Exception as e:
            logger.error(f"LLM domain detection failed, using pattern fallback: {str(e)}")
            domains = _derive_domain_patterns(job.get('job_description', ''))
        
        # Dynamically analyze the cover letter's style and tone
        cover_letter = job.get('cover_letter', '')
        if cover_letter:
            style_analysis = await _analyze_style_and_tone(cover_letter, client)
        else:
            style_analysis = _default_style_analysis()

        # Construct content with proper formatting
        content = f"""
        Job Title: {job.get('title', 'N/A')}
        Status: {raw_status}
        Proposal Date: {job.get('proposal_date', 'N/A')}
        
        Job Description:
        {job.get('job_description', 'N/A')}
        
        My Proposal:
        {job.get('cover_letter', 'N/A')}
        """

        # Build comprehensive metadata
        metadata = {
            # Basic job info
            "type": "job_history",
            "title": str(job.get('title', '')),
            "status": raw_status,
            "status_type": status_type,
            "was_hired": status_type == 'hired',
            "is_closed": status_type in ['closed', 'hired'],  # Consider hired jobs also closed
            "proposal_date": str(job.get('proposal_date', '')),
            "job_url": str(job.get('url', '')),
            
            # Client information
            "client_location": str(client_info.get('location', '')),
            "client_city": str(client_info.get('city', '')),
            "client_company_size": str(client_info.get('company_size', '')),
            "client_member_since": str(client_info.get('member_since', '')),
            "client_total_spent": total_spent,
            "client_avg_hourly_rate": avg_hourly,
            "client_total_hours": total_hours,
            "client_hire_rate": hire_rate,
            "client_has_open_jobs": has_open_jobs,
            "client_total_hires": total_hires,
            "client_active_hires": active_hires,
            "client_rating": rating,
            "client_review_score": review_score,
            "client_review_count": review_count,
            "client_jobs_posted": extract_number(client_info.get('jobs_posted', ''), r'(\d+)'),
            "client_payment_verified": "verified" in str(client_info.get('payment_verified', '')).lower(),
            
            # Proposal metrics
            "has_cover_letter": bool(job.get('cover_letter')),
            "cover_letter_length": len(str(job.get('cover_letter', ''))),
            "job_description_length": len(str(job.get('job_description', ''))),
            
            # Analysis metrics
            "proposal_complexity": len(str(job.get('cover_letter', '')).split()),
            "job_complexity": len(str(job.get('job_description', '')).split()),
            
            # Enhanced domain information
            "domains": domains,
            "primary_domain": domains[0] if domains else "general",
            "is_multi_domain": len(domains) > 1,
            "domain_confidence": "llm" if len(domains) > 0 else "pattern",
            
            # Style and tone analysis
            "overall_tone": style_analysis.get("overall_tone", "professional"),
            "technical_depth": style_analysis.get("technical_depth", "intermediate"),
            "voice_style": style_analysis.get("voice_style", "direct"),
            "raw_style_analysis": style_analysis.get("raw_analysis", ""),
            
            # Timestamps
            "processed_at": int(time.time()),
            "qa": job.get('qa_section', [])
        }

        # Generate unique ID
        doc_id = generate_unique_doc_id(
            "job",
            f"{job['title']}_{raw_status}"
        )

        docs.append({
            "id": doc_id,
            "content": content,
            "metadata": metadata
        })
        
        logger.info(f"Processed job: {metadata['title']} (ID: {doc_id}) - Status: {status_type}")

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
    
    # Load only jobs data
    print("Loading job history documents...")
    with open("data_processing/raw_data/jobs_data.json", "r", encoding="utf-8") as f:
        jobs_data = json.load(f)
    
    # Process only job history documents
    all_docs = asyncio.run(process_jobs_history(jobs_data))
    
    # Upload to single index with force_recreate=True
    total_chunks = process_and_upload_documents(all_docs, INDEX_NAME)

    print("\n=== Processing Complete ===")
    print(f"Total documents: {len(all_docs)}")
    print(f"Total chunks: {total_chunks}")

if __name__ == "__main__":
    main() 