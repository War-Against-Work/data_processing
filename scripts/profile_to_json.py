import os
import json
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

def read_profile_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def process_profile_with_openai(content):
    client = OpenAI()
    
    system_prompt = """You are a data structuring assistant. Parse the freelancer profile data into structured JSON following this exact format:

    {
      "profiles": [
        {
          "title": "string",
          "specialization": "string",
          "hourly_rate": number,
          "overview": "string",
          "key_achievements": ["string"],
          "areas_of_expertise": ["string"],
          "notable_clients": ["string"],
          "approach": "string",
          "key_skills": ["string"],
          "technical_expertise": {
            "languages": ["string"],
            "frameworks": ["string"],
            "tools": ["string"],
            "concepts": ["string"]
          },
          "notable_projects": ["string"],
          "skills_and_expertise": {
            "databases": ["string"],
            "development_languages": ["string"],
            "development_deliverables": ["string"],
            "development_skills": ["string"],
            "web_servers": ["string"],
            "other_skills": ["string"]
          }
        }
      ]
    }

    Important parsing rules:
    1. Extract all profile variations (Digital Product Strategist, AI Solutions Architect, Full Stack Developer)
    2. Convert hourly rates to numbers without currency symbols
    3. Split lists into arrays
    4. Preserve original formatting in text fields
    5. Group skills into appropriate categories as shown in the profile
    6. Include all technical expertise when available
    7. Parse complete overview and approach sections
    8. Maintain original ordering of skills within categories"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Parse this profile data into structured JSON: {content}"}
            ],
            temperature=0.1,
            max_tokens=4000,
            stream=True
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
        
        # Parse the complete response
        result = json.loads(collected_response)
        print(f"\nSuccessfully processed {len(result['profiles'])} profiles")
        
        return json.dumps(result, indent=2, ensure_ascii=False)
    
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
    
    input_file = "profiles.md"
    output_file = "profile_data.json"
    
    print("Starting profile processing...")
    
    # Read the input file
    content = read_profile_file(input_file)
    print(f"Read input file: {input_file}")
    
    # Process with OpenAI
    json_output = process_profile_with_openai(content)
    
    if json_output:
        # Save the structured JSON
        save_json_output(json_output, output_file)
        print("Processing completed successfully!")
    else:
        print("Failed to process profiles")

if __name__ == "__main__":
    main() 