import json
from pathlib import Path

def validate_jobs_data():
    """Validate and fix the structure of jobs_data.json"""
    
    # Required fields and their default values
    required_fields = {
        'title': "",
        'status': "",
        'proposal_date': "",
        'job_description': "",
        'cover_letter': "",
        'qa_section': [],
        'client_info': {},
        'outcome': "",
        'url': ""
    }
    
    # Required client_info fields and their default values
    required_client_fields = {
        'location': "",
        'city': "",
        'jobs_posted': "",
        'hire_info': "",
        'total_spent': "",
        'hires_info': "",
        'avg_hourly_rate': "",
        'hours': "",
        'member_since': "",
        'rating': "",
        'reviews': "",
        'payment_verified': ""
    }

    try:
        # Load the data
        input_path = Path("data_processing/raw_data/jobs_data.json")
        with open(input_path, 'r') as f:
            jobs_data = json.load(f)

        print(f"\nValidating {len(jobs_data)} job records...")
        fixed_count = 0
        
        # Validate and fix each job
        for i, job in enumerate(jobs_data):
            modified = False
            
            # Ensure all required fields exist
            for field, default_value in required_fields.items():
                if field not in job:
                    job[field] = default_value
                    modified = True
                    print(f"Added missing field '{field}' to job {i}")
                    
                # Special handling for client_info
                if field == 'client_info' and isinstance(job[field], dict):
                    for client_field, client_default in required_client_fields.items():
                        if client_field not in job[field]:
                            job[field][client_field] = client_default
                            modified = True
                            print(f"Added missing client_info field '{client_field}' to job {i}")
                
                # Special handling for qa_section
                if field == 'qa_section' and not isinstance(job[field], list):
                    job[field] = []
                    modified = True
                    print(f"Fixed qa_section type in job {i}")
                    
            if modified:
                fixed_count += 1

        # Save the validated data
        with open(input_path, 'w') as f:
            json.dump(jobs_data, f, indent=2)
            
        print(f"\nValidation complete:")
        print(f"Total jobs processed: {len(jobs_data)}")
        print(f"Jobs fixed: {fixed_count}")
        print(f"Data saved to {input_path}")

    except Exception as e:
        print(f"Error during validation: {str(e)}")
        raise

if __name__ == "__main__":
    validate_jobs_data() 