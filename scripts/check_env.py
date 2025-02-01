import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Print relevant environment variables
print("\nCurrent Environment Variables:")
print("-" * 50)
print("OpenAI API Key:", os.getenv('OPENAI_API_KEY'))
print("Pinecone API Key:", os.getenv('PINECONE_API_KEY'))
print("Pinecone Environment:", os.getenv('PINECONE_ENV'))
print("Index Name:", os.getenv('INDEX_NAME'))
print("-" * 50) 