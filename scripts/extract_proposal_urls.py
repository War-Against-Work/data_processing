import re
import json
import os

def extract_urls_from_log(log_text):
    """Extract proposal URLs and their outcomes from log output and save to proposal_urls.json"""
    # Regular expression to match both URL and status
    pattern = r'Found proposal link: (https://www\.upwork\.com/nx/proposals/\d+) \(Status: ([^)]+)\)'
    
    # Find all matches and extract URLs with their outcomes
    proposals = [
        {
            'url': match.group(1),
            'outcome': match.group(2)
        }
        for match in re.finditer(pattern, log_text)
    ]
    
    # Remove duplicates while maintaining order
    seen = set()
    unique_proposals = []
    for prop in proposals:
        if prop['url'] not in seen:
            seen.add(prop['url'])
            unique_proposals.append(prop)
    
    print(f"Extracted {len(unique_proposals)} unique proposals")
    
    # Save to file
    try:
        os.makedirs('data_processing/raw_data', exist_ok=True)
        with open('data_processing/raw_data/proposal_urls.json', 'w') as f:
            json.dump(unique_proposals, f, indent=2)
        print("Saved proposals to proposal_urls.json")
    except Exception as e:
        print(f"Error saving proposals: {str(e)}")
    
    return unique_proposals

if __name__ == "__main__":
    # Read the log file and extract URLs with outcomes
    with open('data_processing/raw_data/log.txt', 'r') as f:
        proposals = extract_urls_from_log(f.read())
    print(f"Found {len(proposals)} proposals to process") 