import os
import json
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

def read_jobs_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def chunk_content(content, max_chars=8000):
    """Split content into chunks that won't exceed token limits"""
    jobs = content.split("___")
    chunks = []
    current_chunk = []
    current_length = 0
    
    for job in jobs:
        if len(job.strip()) == 0:
            continue
        
        job_length = len(job)
        if current_length + job_length > max_chars and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [job]
            current_length = job_length
        else:
            current_chunk.append(job)
            current_length += job_length
    
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    
    print(f"Split content into {len(chunks)} chunks")
    return chunks

def process_jobs_with_openai(content):
    client = OpenAI()
    
    system_prompt = """You are a data structuring assistant. Parse freelance job history data into structured JSON following this exact format:

    {
      "jobs": [
        {
          "uid": "string (generate a unique identifier combining job title and date, e.g. 'web-dev-2023-04-22')",
          "title": "string",
          "date_started": "YYYY-MM-DD or null",
          "date_ended": "YYYY-MM-DD or null",
          "date_posted": "YYYY-MM-DD or null",
          "time_spent_hours": number or null,
          "hourly_rate": number or null,
          "total_earned": number or null,
          "job_feedback": {
            "client_feedback_given": {
              "rating": number or null,
              "text": "string"
            } or "No feedback given",
            "freelancer_feedback_given": {
              "rating": number or null,
              "text": "string"
            } or null
          },
          "job_details": {
            "job_description": "string",
            "job_type": "Hourly" or "Fixed price",
            "estimated_hours_per_week": "string or null",
            "estimated_duration": "string or null",
            "experience_level": "string or null",
            "budget": number or null,
            "hourly_range": "string or null",
            "company_description": "string or null"
          },
          "client_info": {
            "rating": number or null,
            "reviews_count": number or null,
            "location": "string or null",
            "total_spent": number or null,
            "hires": number or null
          },
          "cover_letter": "string or null"
        }
      ]
    }

    Important parsing rules:
    1. Generate a unique uid by combining job title and date in kebab-case (lowercase with hyphens)
       - Use date_started if available, otherwise date_posted
       - Remove special characters and spaces from title
       - Example: "Web Developer needed" started on "2023-04-22" becomes "web-developer-needed-2023-04-22"
    2. Extract full text of feedback and cover letters
    3. Convert all monetary values to numbers without currency symbols
    4. Parse dates into YYYY-MM-DD format
    5. Include all client information when available
    6. Preserve original formatting in text fields
    7. Set fields to null when data is not available
    8. For client_feedback_given, use "No feedback given" string when explicitly stated
    9. Extract complete job descriptions including any additional details
    10. Include hourly range and budget information when available
    11. Parse company/business descriptions when provided."""

    try:
        # Split content into manageable chunks
        chunks = chunk_content(content)
        all_jobs = []
        
        for i, chunk in enumerate(chunks, 1):
            print(f"\nProcessing chunk {i} of {len(chunks)}...")
            
            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Parse these jobs into structured JSON: {chunk}"}
                ],
                temperature=0.1,
                max_tokens=4000,
                stream=True  # Enable streaming
            )
            
            # Initialize collected response
            collected_response = ""
            print("Receiving response: ", end="", flush=True)
            
            # Process the stream
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_response += content
                    print(".", end="", flush=True)
            
            print("\nParsing response...")
            
            try:
                # Parse the complete response
                chunk_result = json.loads(collected_response)
                if 'jobs' in chunk_result:
                    jobs_in_chunk = chunk_result['jobs']
                    all_jobs.extend(jobs_in_chunk)
                    print(f"Successfully processed {len(jobs_in_chunk)} jobs from chunk {i}")
                else:
                    print(f"Warning: No jobs found in chunk {i}")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from chunk {i}: {e}")
                continue
            
        # Combine all jobs into final structure
        final_json = {
            "total_jobs": len(all_jobs),
            "jobs": all_jobs
        }
        
        print(f"\nTotal jobs processed: {len(all_jobs)}")
        return json.dumps(final_json, indent=2, ensure_ascii=False)
    
    except Exception as e:
        print(f"\nError processing with OpenAI: {e}")
        return None

def save_json_output(json_content, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(json_content)
    print(f"\nSaved output to {output_file}")

def main():
    # Load environment variables
    load_dotenv()
    
    input_file = "successful_jobs.md"
    output_file = "jobs_history.json"
    
    print("Starting job processing...")
    
    # Read the input file
    content = read_jobs_file(input_file)
    print(f"Read input file: {input_file}")
    
    # Process with OpenAI
    json_output = process_jobs_with_openai(content)
    
    if json_output:
        # Save the structured JSON
        save_json_output(json_output, output_file)
        print("Processing completed successfully!")
    else:
        print("Failed to process jobs")

if __name__ == "__main__":
    main() 