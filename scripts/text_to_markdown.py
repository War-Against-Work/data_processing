import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def split_into_jobs(text: str, jobs_per_chunk: int = 5) -> list[str]:
    """
    Split text into chunks based on job entries.
    Each chunk will contain up to jobs_per_chunk number of jobs.
    """
    # First, normalize the text to ensure consistent job separators
    lines = text.splitlines()
    job_sections = []
    current_job = []
    
    for line in lines:
        # Check for job title markers - could be # or ## with various titles
        if (line.startswith('# ') or line.startswith('## ')) and any(
            job_indicator in line.lower() for job_indicator in 
            ['needed', 'expert', 'developer', 'designer', 'assistance', 'website', 'help']
        ):
            if current_job:  # Save previous job if exists
                job_sections.append('\n'.join(current_job))
                current_job = []
        current_job.append(line)
    
    # Add the last job
    if current_job:
        job_sections.append('\n'.join(current_job))
    
    print(f"Found {len(job_sections)} job sections")
    
    # Group jobs into chunks
    chunks = []
    for i in range(0, len(job_sections), jobs_per_chunk):
        chunk_jobs = job_sections[i:i + jobs_per_chunk]
        chunk = '\n\n'.join(chunk_jobs)  # Add extra newline between jobs
        if chunk.strip():
            chunks.append(chunk)
    
    print(f"Created {len(chunks)} chunks of up to {jobs_per_chunk} jobs each")
    return chunks

def split_into_profiles(text: str, profiles_per_chunk: int = 3) -> list[str]:
    """
    Split text into chunks based on profile entries.
    Each chunk will contain up to profiles_per_chunk number of profiles.
    """
    # Split on profile sections (assuming they start with # Name or ## Name)
    profile_sections = []
    current_section = []
    
    for line in text.splitlines(True):  # keepends=True to preserve newlines
        if (line.startswith('# ') or line.startswith('## ')) and current_section:
            profile_sections.append(''.join(current_section))
            current_section = []
        current_section.append(line)
    
    # Add the last section
    if current_section:
        profile_sections.append(''.join(current_section))
    
    print(f"Found {len(profile_sections)} profile sections")
    
    # Group profiles into chunks
    chunks = []
    for i in range(0, len(profile_sections), profiles_per_chunk):
        chunk = ''.join(profile_sections[i:i + profiles_per_chunk])
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
    
    print(f"Created {len(chunks)} chunks of {profiles_per_chunk} profiles each")
    return chunks

def convert_text_to_markdown(input_text: str, max_tokens: int = 4000) -> str:
    """
    Convert plain text to properly formatted markdown using OpenAI's API
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object" },
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": """You are a markdown formatting expert. Convert the given plain text into properly formatted markdown.
                Follow these rules:
                1. Identify and format headers appropriately (h1, h2, h3)
                2. Format lists properly (bullet points or numbered lists)
                3. Identify and format code blocks
                4. Add proper spacing between sections
                5. Format any links or references properly
                6. Identify and format any tables if present
                7. Preserve any existing section breaks and structure
                Return the result in JSON format with a single key 'markdown_content'."""
            },
            {
                "role": "user",
                "content": input_text
            }
        ]
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        return result['markdown_content']
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error processing response: {e}")
        return input_text

def process_file(input_path: str, output_path: str = None):
    """
    Process a single file and convert it to markdown
    """
    print(f"\nProcessing {input_path}...")
    
    # Read input file
    with open(input_path, 'r', encoding='utf-8') as file:
        input_text = file.read()
    
    print(f"Read {len(input_text)} characters")
    
    # Determine file type and split accordingly
    file_name = Path(input_path).stem
    if 'profiles' in file_name.lower():
        chunks = split_into_profiles(input_text, profiles_per_chunk=3)
        chunk_type = "profiles"
    else:
        chunks = split_into_jobs(input_text, jobs_per_chunk=5)
        chunk_type = "jobs"
    
    # Process each chunk
    markdown_chunks = []
    for i, chunk in enumerate(chunks, 1):
        print(f"\nProcessing chunk {i} of {len(chunks)} (containing up to {3 if chunk_type == 'profiles' else 5} {chunk_type})...")
        print(f"Chunk size: {len(chunk)} characters")
        markdown_chunk = convert_text_to_markdown(chunk)
        markdown_chunks.append(markdown_chunk)
    
    # Combine chunks with newlines
    markdown_content = '\n'.join(markdown_chunks)
    
    # If no output path specified, create one
    if output_path is None:
        input_path = Path(input_path)
        output_path = input_path.parent / f"{input_path.stem}.md"
    
    # Write the markdown content
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(markdown_content)
    
    print(f"\nConverted {input_path} to {output_path}")

def main():
    # Directory paths
    raw_data_dir = Path("data_processing/raw_data")
    processed_dir = Path("data_processing/processed_data")
    
    # Create processed directory if it doesn't exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all .txt files in raw_data
    for txt_file in raw_data_dir.glob("*.txt"):
        output_path = processed_dir / f"{txt_file.stem}.md"
        process_file(str(txt_file), str(output_path))

if __name__ == "__main__":
    main() 