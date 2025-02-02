# Data Processing

This directory contains scripts for scraping Upwork data and processing embeddings.

## Directory Structure
- `scripts/`: Contains the main processing scripts
  - `upwork_scraper.py`: Scrapes data from Upwork
  - `process_embeddings.py`: Processes and generates embeddings
  - `test_chrome_connection.py`: Tests Chrome/Selenium connection
  - `tests/`: Test files for the scripts
- `raw_data/`: Storage for raw scraped data
- `processed_data/`: Storage for processed embeddings and data

## Usage
1. Run the scraper: `python scripts/upwork_scraper.py`
2. Process embeddings: `python scripts/process_embeddings.py` 