import os
import json
import argparse
import time
import asyncio
import logging
import re
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
import tiktoken
import openai
from tqdm import tqdm
from dotenv import load_dotenv
import hashlib
from colorama import init, Fore, Style
from local_embeddings import LocalEmbeddingService

# Initialize colorama for Windows compatibility
init()

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        # Add timing information
        if not hasattr(record, 'elapsed'):
            record.elapsed = ''
        
        # Add color based on level
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
            
        # Color the message based on type
        if hasattr(record, 'msg_type'):
            if record.msg_type == 'uuid':
                record.msg = f"{Fore.CYAN}{record.msg}{Style.RESET_ALL}"
            elif record.msg_type == 'embedding':
                record.msg = f"{Fore.MAGENTA}{record.msg}{Style.RESET_ALL}"
            elif record.msg_type == 'llm':
                record.msg = f"{Fore.YELLOW}{record.msg}{Style.RESET_ALL}"
                
        return super().format(record)

# Configure logging with new formatter
def setup_logging():
    formatter = ColoredFormatter(
        '%(asctime)s [%(levelname)s] %(message)s %(elapsed)s'
    )
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    return logger

logger = setup_logging()

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("INDEX_NAME", "upwork-v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-ada-002")
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
BATCH_SIZE = 100  # Not used in one-by-one upsert

# Update these constants at the top
LOCAL_EMBED_MODEL = "BAAI/bge-large-en-v1.5"  # This produces 1024-dim vectors that we'll pad to 1536
VECTOR_DIM = 1536  # Match existing Pinecone index dimension

client = openai.OpenAI()

# At the top of the file, after other imports
logging.getLogger('pinecone').setLevel(logging.INFO)

# Initialize embedding service
local_embedding_service = LocalEmbeddingService() if USE_LOCAL_EMBEDDINGS else None

def clean_metadata_value(value):
    """Clean metadata values for Pinecone storage.
    
    Args:
        value: Any metadata value that needs cleaning
        
    Returns:
        Cleaned value suitable for Pinecone storage
    """
    if value is None:
        return None
        
    if isinstance(value, (int, float, bool)):
        return value
        
    if isinstance(value, dict):
        return {k: clean_metadata_value(v) for k, v in value.items()}
        
    if isinstance(value, list):
        return [clean_metadata_value(item) for item in value]
        
    # Convert any other types to string
    try:
        cleaned = str(value).strip()
        # Remove any null-like strings
        if cleaned.lower() in ('null', 'none', 'undefined', 'nan'):
            return None
        return cleaned
    except:
        return None

def get_embeddings_batch(texts: List[str]) -> Optional[List[List[float]]]:
    """Get embeddings using either local model or OpenAI API."""
    if USE_LOCAL_EMBEDDINGS:
        logger.debug("Using local embedding model")
        return local_embedding_service.get_embeddings(texts)
    
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
                logger.error(f"Embedding failed after {max_retries} attempts: {e}")
                return None
            logger.info(f"Retry {attempt + 1}/{max_retries} after error: {e}")
            time.sleep(2 ** attempt)

def tokenize_text(text, encoder):
    return encoder.encode(text, disallowed_special=())

def detokenize_text(tokens, encoder):
    return encoder.decode(tokens)

def chunk_text(text, chunk_size, encoder):
    tokens = tokenize_text(text, encoder)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(detokenize_text(tokens[start:end], encoder))
        start = end
    return chunks

def init_pinecone(force_recreate=False):
    """Initialize Pinecone index with correct dimensions."""
    try:
        logger.info("=== Starting Pinecone Initialization ===")
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Always use 1536 dimensions to match OpenAI's format
        dimension = VECTOR_DIM
        
        existing_indexes = pc.list_indexes()
        logger.info(f"Found existing indexes: {[index.name for index in existing_indexes]}")
        
        if force_recreate:
            if INDEX_NAME in [index.name for index in existing_indexes]:
                logger.info(f"Deleting existing index '{INDEX_NAME}'...")
                pc.delete_index(INDEX_NAME)
                logger.info("Index deletion initiated")
                logger.info("Waiting for deletion to complete...")
                time.sleep(20)
        
        if INDEX_NAME not in [index.name for index in existing_indexes] or force_recreate:
            logger.info(f"Creating new index '{INDEX_NAME}' with dimension {dimension}...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            logger.info("Index creation initiated")
            logger.info("Waiting for index to be ready...")
            time.sleep(20)
            
        logger.info("=== Pinecone Initialization Complete ===")
        return pc.Index(INDEX_NAME)
        
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        raise

def clean_for_embedding(text):
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"[^\x20-\x7E\n\t]", "", text)
    text = re.sub(r"undefined|null|NaN|Infinity|-Infinity", "", text)
    text = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "", text)
    text = re.sub(r"[\{\[\(].*?[\}\]\)]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text_with_overlap(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(encoder.decode(tokens[start:end]))
        start = end - overlap if end < len(tokens) else end
    return chunks

def generate_stable_uuid(doc: Dict[str, Any]) -> str:
    """
    Generate a stable UUID for a document based on unique fields.
    
    Args:
        doc: Document dictionary containing job data
        
    Returns:
        Stable UUID string
    """
    # Use job URL as primary unique identifier, fall back to title + date if no URL
    unique_string = doc.get("url", "") or f"{doc.get('title', '')}_{doc.get('proposal_date', '')}"
    hash_value = hashlib.md5(unique_string.encode("utf8")).hexdigest()
    return f"job_{hash_value[:12]}"  # Use first 12 chars for readability

def _derive_domain_patterns(job_description: str) -> List[str]:
    if not job_description:
        return ["general"]
    text = job_description.lower()
    domains = set()
    basic_domains = {
        "finance": ["finance", "bank", "invest", "trading", "accounting"],
        "healthcare": ["health", "medical", "clinic", "patient"],
        "technology": ["tech", "software", "data", "cloud", "ai"],
        "marketing": ["market", "brand", "content", "seo", "advertising"],
        "education": ["education", "learn", "teach", "training", "course"],
        "ecommerce": ["commerce", "shop", "retail", "marketplace"]
    }
    for domain, keywords in basic_domains.items():
        if any(keyword in text for keyword in keywords):
            domains.add(domain)
    if any(keyword in text for keyword in ["web", "website", "frontend", "backend"]):
        domains.add("web_development")
    if any(keyword in text for keyword in ["mobile", "ios", "android", "app"]):
        domains.add("mobile_development")
    return list(domains) if domains else ["general"]

async def _derive_domain_llm(job_description: str, client: openai.OpenAI) -> List[str]:
    try:
        messages = [
            {"role": "developer", "content": (
                "You are a domain classification expert.\n"
                "Given a job description, identify the most relevant business and technical domains.\n"
                "Return a JSON object with a 'domains' array containing relevant domain strings."
            )},
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
            if not domains:
                logger.warning("LLM returned empty domains list, using pattern fallback")
                return _derive_domain_patterns(job_description)
            logger.info(f"LLM detected domains: {domains}")
            return domains
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse LLM domain response: {e}, using pattern fallback")
            return _derive_domain_patterns(job_description)
    except Exception as e:
        logger.error(f"LLM domain analysis failed: {e}, using pattern fallback")
        return _derive_domain_patterns(job_description)

async def _analyze_style_and_tone(text: str, client: openai.OpenAI) -> dict:
    try:
        messages = [
            {"role": "developer", "content": (
                "You are an expert proposal analyst.\n"
                "Analyze the provided text and assess its tone, technical depth, and voice style.\n"
                "Return a JSON object with keys: overall_tone, technical_depth, voice_style, raw_analysis."
            )},
            {"role": "user", "content": f"Analyze this proposal text:\n{text}"}
        ]
        completion = client.chat.completions.create(
            model="o3-mini",
            messages=messages,
            reasoning_effort="medium",
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
        logger.error(f"Dynamic style analysis failed: {e}")
        return _default_style_analysis()

def _default_style_analysis() -> dict:
    return {
        "overall_tone": "professional",
        "technical_depth": "intermediate",
        "voice_style": "direct",
        "raw_analysis": ""
    }

async def process_jobs_history(jobs_data):
    docs = []
    for job in jobs_data:
        raw_status = job.get("outcome", job.get("status", "")).lower()
        status_mapping = {"hired": "hired", "job is closed": "closed"}
        is_date = bool(re.match(r"[A-Za-z]+ \d{1,2}, \d{4}", raw_status))
        status_type = "date" if is_date else next((v for k, v in status_mapping.items() if k in raw_status), "other")
        client_info = job.get("client_info", {})

        def extract_number(text, pattern, convert_k=False):
            if not text:
                return None
            match = re.search(pattern, text)
            if not match:
                return None
            try:
                value = float(match.group(1))
                if convert_k and "K" in text:
                    value *= 1000
                return value
            except (ValueError, TypeError):
                return None

        total_spent = extract_number(client_info.get("total_spent", ""), r"\$?([\d.]+)", convert_k=True)
        avg_hourly = extract_number(client_info.get("avg_hourly_rate", ""), r"\$?([\d.]+)")
        total_hours = extract_number(client_info.get("hours", ""), r"([\d.]+)")
        hire_info = client_info.get("hire_info", "")
        hire_rate = extract_number(hire_info, r"(\d+)%")
        has_open_jobs = "open job" in hire_info.lower()
        hires_info = client_info.get("hires_info", "") or ""
        hires_match = re.search(r"(\d+) hires?, (\d+) active", hires_info)
        total_hires = int(hires_match.group(1)) if hires_match else 0
        active_hires = int(hires_match.group(2)) if hires_match else 0
        rating = extract_number(client_info.get("rating", ""), r"([\d.]+)")
        reviews_match = re.search(r"([\d.]+) of (\d+) reviews", client_info.get("reviews", "") or "")
        review_score = float(reviews_match.group(1)) if reviews_match else None
        review_count = int(reviews_match.group(2)) if reviews_match else 0

        try:
            domains = await _derive_domain_llm(job.get("job_description", ""), client)
        except Exception as e:
            logger.error(f"LLM domain detection failed: {e}, using pattern fallback")
            domains = _derive_domain_patterns(job.get("job_description", ""))

        cover_letter = job.get("cover_letter", "")
        if cover_letter:
            style_analysis = await _analyze_style_and_tone(cover_letter, client)
        else:
            style_analysis = _default_style_analysis()

        content = (
            f"Job Title: {job.get('title', 'N/A')}\n"
            f"Status: {raw_status}\n"
            f"Proposal Date: {job.get('proposal_date', 'N/A')}\n\n"
            f"Job Description:\n{job.get('job_description', 'N/A')}\n\n"
            f"My Proposal:\n{job.get('cover_letter', 'N/A')}\n"
        )

        metadata = {
            "type": "job_history",
            "title": str(job.get("title", "")),
            "status": raw_status,
            "status_type": status_type,
            "was_hired": status_type == "hired",
            "is_closed": status_type in ["closed", "hired"],
            "proposal_date": str(job.get("proposal_date", "")),
            "job_url": str(job.get("url", "")),
            "client_location": str(client_info.get("location", "")),
            "client_city": str(client_info.get("city", "")),
            "client_company_size": str(client_info.get("company_size", "")),
            "client_member_since": str(client_info.get("member_since", "")),
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
            "client_jobs_posted": extract_number(client_info.get("jobs_posted", ""), r"(\d+)"),
            "client_payment_verified": "verified" in str(client_info.get("payment_verified", "")).lower(),
            "has_cover_letter": bool(job.get("cover_letter")),
            "cover_letter_length": len(str(job.get("cover_letter", ""))),
            "job_description_length": len(str(job.get("job_description", ""))),
            "proposal_complexity": len(str(job.get("cover_letter", "")).split()),
            "job_complexity": len(str(job.get("job_description", "")).split()),
            "domains": domains,
            "primary_domain": domains[0] if domains else "general",
            "is_multi_domain": len(domains) > 1,
            "domain_confidence": "llm" if domains else "pattern",
            "overall_tone": style_analysis.get("overall_tone", "professional"),
            "technical_depth": style_analysis.get("technical_depth", "intermediate"),
            "voice_style": style_analysis.get("voice_style", "direct"),
            "raw_style_analysis": style_analysis.get("raw_analysis", ""),
            "processed_at": int(time.time()),
            "qa": job.get("qa_section", [])
        }

        doc_id = generate_stable_uuid(job)
        docs.append({
            "id": doc_id,
            "content": content,
            "metadata": metadata
        })
        logger.info(f"Processed job: {metadata['title']} (ID: {doc_id}) - Status: {status_type}")
    return docs

def process_profile_data(profile_data):
    docs = []
    for profile in profile_data["profiles"]:
        unique_id = generate_stable_uuid(profile)
        profile_text = (
            f"Title: {profile['title']}\n"
            f"Specialization: {profile['specialization']}\n"
            f"Hourly Rate: ${profile['hourly_rate']}\n\n"
            f"Overview: {profile['overview']}\n\n"
            f"Key Achievements: {' '.join(profile.get('key_achievements', []))}\n\n"
            f"Areas of Expertise: {' '.join(profile.get('areas_of_expertise', []))}\n\n"
            f"Notable Clients: {' '.join(profile.get('notable_clients', []))}\n\n"
            f"Approach: {profile['approach']}\n\n"
            f"Technical Skills: {', '.join(profile.get('technical_expertise', {}).get('languages', []) + profile.get('technical_expertise', {}).get('frameworks', []) + profile.get('technical_expertise', {}).get('tools', []))}"
        )
        metadata = {
            "type": "profile",
            "source": "profile_data.json",
            "doc_id": unique_id,
            "title": profile["title"],
            "specialization": profile["specialization"],
            "hourly_rate": float(profile["hourly_rate"]),
            "has_key_achievements": bool(profile.get("key_achievements")),
            "notable_clients_count": len(profile.get("notable_clients", [])),
            "skills_count": len(profile.get("technical_expertise", {}).get("languages", []) +
                                profile.get("technical_expertise", {}).get("frameworks", []) +
                                profile.get("technical_expertise", {}).get("tools", [])),
            "languages": profile.get("technical_expertise", {}).get("languages", []),
            "frameworks": profile.get("technical_expertise", {}).get("frameworks", []),
            "tools": profile.get("technical_expertise", {}).get("tools", []),
            "processed_at": int(time.time())
        }
        docs.append({
            "id": unique_id,
            "content": profile_text,
            "metadata": metadata
        })
    return docs

def flatten_client_info(client_info):
    """Flatten client info dictionary into simple key-value pairs."""
    if not client_info:
        return {}
        
    flattened = {
        "client_location": str(client_info.get("location", "")),
        "client_city": str(client_info.get("city", "")),
        "client_company_size": str(client_info.get("company_size", "")),
        "client_member_since": str(client_info.get("member_since", "")),
        "client_payment_verified": "verified" in str(client_info.get("payment_verified", "")).lower(),
        "client_jobs_posted": float(re.search(r"(\d+)", client_info.get("jobs_posted", "0") or "0").group(1)) if re.search(r"(\d+)", client_info.get("jobs_posted", "0") or "0") else 0,
        "client_hire_rate": float(re.search(r"(\d+)%", client_info.get("hire_info", "") or "0").group(1))/100 if re.search(r"(\d+)%", client_info.get("hire_info", "") or "0") else None,
        "client_total_spent": float(re.search(r"\$?([\d.]+)K?", client_info.get("total_spent", "0") or "0").group(1)) if re.search(r"\$?([\d.]+)K?", client_info.get("total_spent", "0") or "0") else 0,
        "client_rating": float(re.search(r"([\d.]+)", client_info.get("rating", "0") or "0").group(1)) if re.search(r"([\d.]+)", client_info.get("rating", "0") or "0") else None,
        "client_review_count": int(re.search(r"[\d.]+ of (\d+) reviews", client_info.get("reviews", "") or "0").group(1)) if re.search(r"[\d.]+ of (\d+) reviews", client_info.get("reviews", "") or "0") else 0,
        "client_review_score": float(re.search(r"([\d.]+) of \d+ reviews", client_info.get("reviews", "") or "0").group(1)) if re.search(r"([\d.]+) of \d+ reviews", client_info.get("reviews", "") or "0") else None,
    }
    
    return {k: v for k, v in flattened.items() if v is not None}

async def process_and_upload_single_doc(doc, index, client):
    """Process and upload a single document with separate embeddings for job and proposal."""
    start_time = time.time()
    
    # Log embedding service being used
    if not hasattr(process_and_upload_single_doc, '_logged_embedding_service'):
        logger.info(f"Using {'local GPU' if USE_LOCAL_EMBEDDINGS else 'OpenAI API'} for embeddings")
        process_and_upload_single_doc._logged_embedding_service = True
    
    # Generate stable UUID for the document
    doc_uuid = generate_stable_uuid(doc)
    rfp_id = f"{doc_uuid}_rfp"
    proposal_id = f"{doc_uuid}_proposal"
    
    # Add separator line for clarity
    logger.info("=" * 80)
    logger.info(f"Processing document: {doc.get('title', '')}", extra={
        'msg_type': 'title',
        'elapsed': f"[{time.time() - start_time:.2f}s]"
    })
    logger.info(f"Document UUID: {doc_uuid}", extra={
        'msg_type': 'uuid',
        'elapsed': f"[{time.time() - start_time:.2f}s]"
    })
    
    # Check if document already exists
    try:
        fetched = index.fetch(ids=[rfp_id, proposal_id])
        if any(fetched.get("vectors", {}).keys()):
            logger.info(f"Skipping {doc_uuid} (already exists)", extra={
                'msg_type': 'skip',
                'elapsed': f"[{time.time() - start_time:.2f}s]"
            })
            return 0
    except Exception as e:
        logger.error(f"Error checking existence: {e}", extra={
            'elapsed': f"[{time.time() - start_time:.2f}s]"
        })
        return 0

    # Process with LLM
    llm_start = time.time()
    try:
        domains = await _derive_domain_llm(doc.get("job_description", ""), client)
        logger.info(f"Domains detected: {domains}", extra={
            'msg_type': 'llm',
            'elapsed': f"[{time.time() - llm_start:.2f}s]"
        })
        
        style_start = time.time()
        style_analysis = await _analyze_style_and_tone(doc.get("cover_letter", ""), client) if doc.get("cover_letter") else _default_style_analysis()
        logger.info(f"Style analysis completed", extra={
            'msg_type': 'llm',
            'elapsed': f"[{time.time() - style_start:.2f}s]"
        })
    except Exception as e:
        logger.error(f"LLM processing failed: {e}", extra={
            'elapsed': f"[{time.time() - start_time:.2f}s]"
        })
        return 0

    # Prepare text content
    rfp_text = doc.get("job_description", "").strip()
    proposal_text = doc.get("cover_letter", "").strip()

    if not rfp_text or not proposal_text:
        logger.warning(f"Missing required content for {doc_uuid}")
        return 0

    # Base metadata common to both embeddings
    base_metadata = {
        "doc_uuid": doc_uuid,
        "doc_category": "job_history",
        "search_category": "rfp_similarity",
        "embedding_type": "rfp",  # Will be overridden for proposal
        "title": str(doc.get("title", "")),
        "status": doc.get("outcome", doc.get("status", "N/A")),
        "status_type": "date" if re.match(r"[A-Za-z]+ \d{1,2}, \d{4}", doc.get("outcome", doc.get("status", "N/A")).lower()) 
                    else next((v for k, v in {"hired": "hired", "job is closed": "closed"}.items() 
                              if k in doc.get("outcome", doc.get("status", "N/A")).lower()), "other"),
        "was_hired": doc.get("outcome", doc.get("status", "N/A")).lower() == "hired",
        "is_closed": doc.get("outcome", doc.get("status", "N/A")).lower() in ["closed", "hired"],
        "proposal_date": str(doc.get("proposal_date", "")),
        "job_url": str(doc.get("url", "")),
        **flatten_client_info(doc.get("client_info", {})),
        "domains": domains,
        "primary_domain": domains[0] if domains else "general",
        "is_multi_domain": len(domains) > 1,
        "domain_confidence": "llm" if domains else "pattern",
        "processed_at": int(time.time())
    }

    # Create specific metadata for RFP
    rfp_metadata = {
        **base_metadata,
        "embedding_type": "rfp",  # Explicitly set type
        "text": rfp_text,
        "content_length": len(rfp_text),
        "word_count": len(rfp_text.split()),
        "has_requirements": "requirements" in rfp_text.lower(),
        "has_budget": any(term in rfp_text.lower() for term in ["budget", "rate", "salary", "compensation"])
    }

    # Create specific metadata for proposal
    proposal_metadata = {
        **base_metadata,
        "embedding_type": "proposal",  # Explicitly set type
        "text": proposal_text,
        "overall_tone": style_analysis.get("overall_tone", "professional"),
        "technical_depth": style_analysis.get("technical_depth", "intermediate"),
        "voice_style": style_analysis.get("voice_style", "direct"),
        "raw_style_analysis": style_analysis.get("raw_analysis", ""),
        "search_priority": 1 if base_metadata["was_hired"] else 0
    }

    total_chunks = 0
    
    # Process and upload RFP embedding
    embed_start = time.time()
    try:
        rfp_embedding = get_embeddings_batch([rfp_text])
        if rfp_embedding:
            index.upsert(vectors=[(rfp_id, rfp_embedding[0], rfp_metadata)])
            logger.info(f"[1/2] ✓ RFP embedding upserted      (ID: {rfp_id})", extra={
                'msg_type': 'embedding',
                'elapsed': f"[{time.time() - embed_start:.2f}s]"
            })
            total_chunks += 1
    except Exception as e:
        logger.error(f"Failed to upsert RFP embedding: {e}", extra={
            'elapsed': f"[{time.time() - start_time:.2f}s]"
        })
        return 0

    # Process and upload proposal embedding
    embed_start = time.time()
    try:
        proposal_embedding = get_embeddings_batch([proposal_text])
        if proposal_embedding:
            index.upsert(vectors=[(proposal_id, proposal_embedding[0], proposal_metadata)])
            logger.info(f"[2/2] ✓ Proposal embedding upserted (ID: {proposal_id})", extra={
                'msg_type': 'embedding',
                'elapsed': f"[{time.time() - embed_start:.2f}s]"
            })
            total_chunks += 1
    except Exception as e:
        logger.error(f"Failed to upsert proposal embedding: {e}", extra={
            'elapsed': f"[{time.time() - start_time:.2f}s]"
        })
        return 0

    logger.info("-" * 40)
    logger.info(f"✓ Document processing complete", extra={
        'elapsed': f"[Total: {time.time() - start_time:.2f}s]"
    })
    logger.info("-" * 40)
    return total_chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the index")
    args = parser.parse_args()

    logger.info("=== Starting Document Processing ===")
    
    try:
        logger.info("Initializing Pinecone...")
        index = init_pinecone(force_recreate=args.reset)
        logger.info("Pinecone initialization completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        return

    try:
        with open("data_processing/raw_data/jobs_data.json", "r", encoding="utf-8") as f:
            jobs_data = json.load(f)
        logger.info(f"Loaded {len(jobs_data)} jobs from jobs_data.json")

        total_processed = 0
        total_chunks = 0
        
        for job in tqdm(jobs_data, desc="Processing jobs"):
            chunks_processed = asyncio.run(process_and_upload_single_doc(job, index, client))
            if chunks_processed > 0:
                total_processed += 1
                total_chunks += chunks_processed

        logger.info("=== Processing Complete ===")
        logger.info(f"Total documents processed: {total_processed}")
        logger.info(f"Total embeddings created: {total_chunks}")
        logger.info(f"(Includes both job descriptions and proposals)")
    except Exception as e:
        logger.error(f"Error during document processing: {e}")
        raise

if __name__ == "__main__":
    main()
