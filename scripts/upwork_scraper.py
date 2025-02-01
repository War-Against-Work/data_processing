from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from pathlib import Path
import json
import os
from dotenv import load_dotenv
import time
import random

# Load environment variables
load_dotenv()

class UpworkScraper:
    def __init__(self, max_proposals=5):
        print("Attempting to connect to existing Chrome session...")
        chrome_options = Options()
        chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
        chrome_options.add_argument("--disable-notifications")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            print(f"Connected successfully to: {self.driver.current_url}")
            
            self.wait = WebDriverWait(self.driver, 10)
            self.successful_jobs = []
            self.unsuccessful_jobs = []
            self.max_proposals = max_proposals
            self.proposals_found = 0
            
        except Exception as e:
            print(f"\nError connecting to Chrome session: {str(e)}")
            raise

    def human_delay(self, min_seconds=1, max_seconds=3):
        """Add a random delay to simulate human behavior"""
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)
        
    def wait_for_cloudflare(self, timeout=30):
        """Wait for Cloudflare verification to complete"""
        print("Waiting for Cloudflare verification...")
        try:
            # Wait for either the username field or password field to appear
            self.wait.until(
                EC.any_of(
                    EC.presence_of_element_located((By.ID, "login_username")),
                    EC.presence_of_element_located((By.ID, "login_password"))
                )
            )
            return True
        except TimeoutException:
            return False

    def login(self):
        """Log into Upwork using credentials from environment variables"""
        print("Logging in...")
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.driver.get("https://www.upwork.com/login")
                self.human_delay(2, 4)  # Longer delay after page load
                
                # Handle initial Cloudflare check
                if not self.wait_for_cloudflare():
                    print("Cloudflare challenge detected...")
                    input("Please complete the Cloudflare verification and press Enter to continue...")
                    if not self.wait_for_cloudflare():
                        print("Still on Cloudflare page, retrying...")
                        retry_count += 1
                        continue
                
                # Rest of login process...
                print("Entering username...")
                username = self.wait.until(
                    EC.presence_of_element_located((By.ID, "login_username"))
                )
                self.human_delay(0.5, 1.5)
                username.send_keys(os.getenv('UPWORK_USERNAME'))
                self.human_delay(1, 2)
                
                # Click continue button
                print("Clicking continue...")
                continue_button = self.wait.until(
                    EC.element_to_be_clickable((By.ID, "login_password_continue"))
                )
                self.human_delay(0.5, 1.5)  # Delay before clicking
                continue_button.click()
                
                # Wait for human verification after username
                input("Complete any human verification if needed and press Enter to continue...")
                
                # Wait for password field
                print("Waiting for password field...")
                password = self.wait.until(
                    EC.presence_of_element_located((By.ID, "login_password"))
                )
                self.wait.until(
                    EC.element_to_be_clickable((By.ID, "login_password"))
                )
                self.human_delay(1, 2)  # Delay before typing password
                
                print("Entering password...")
                password.send_keys(os.getenv('UPWORK_PASSWORD'))
                self.human_delay(0.8, 1.5)  # Delay after typing password
                
                # Submit login form
                print("Submitting login form...")
                submit = self.wait.until(
                    EC.element_to_be_clickable((By.ID, "login_control_continue"))
                )
                self.human_delay(0.5, 1.5)  # Delay before final click
                submit.click()
                
                # Wait for human verification after password
                input("Complete any human verification if needed and press Enter to continue...")
                
                print("Waiting for 2FA...")
                input("Please complete 2FA if required and press Enter to continue...")
                
                break  # If we get here, login was successful
                
            except Exception as e:
                print(f"Error during login attempt {retry_count + 1}: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    print("Retrying login...")
                    self.human_delay(5, 10)  # Longer delay between retries
                else:
                    print("Max retries reached. Please try again later.")
                    raise
        
    def get_proposals(self):
        """Navigate to archived proposals page and collect proposal links"""
        print("Fetching proposals...")
        proposals = []
        page = 1
        
        # First check active proposals
        self.driver.get("https://www.upwork.com/nx/find-work/submitted-proposals")
        self.human_delay(2, 4)  # Longer delay after page load
        active_proposals = self._get_proposals_from_current_page()
        proposals.extend(active_proposals)
        print(f"Found {len(active_proposals)} active proposals")
        
        # Then check archived proposals
        self.driver.get("https://www.upwork.com/nx/proposals/archived")
        self.human_delay(2, 4)  # Longer delay after page load
        
        while len(proposals) < self.max_proposals:
            new_proposals = self._get_proposals_from_current_page()
            if not new_proposals:
                break
                
            proposals.extend(new_proposals)
            print(f"Found {len(new_proposals)} proposals on page {page}")
            
            if len(proposals) < self.max_proposals:
                try:
                    next_button = self.driver.find_element(By.CSS_SELECTOR, "[aria-label='Next Page']")
                    if not next_button.is_enabled():
                        break
                    self.human_delay(1, 2)  # Delay before clicking next
                    next_button.click()
                    self.human_delay(2, 3)  # Longer delay after page change
                    page += 1
                except:
                    print("No more pages available")
                    break
        
        return proposals[:self.max_proposals]
    
    def _get_proposals_from_current_page(self):
        """Extract proposal links from current page"""
        try:
            # Wait for the proposals to load
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-test='proposal-card']")))
            
            # Find all proposal cards
            proposal_cards = self.driver.find_elements(By.CSS_SELECTOR, "[data-test='proposal-card']")
            
            # Extract links from cards
            proposals = []
            for card in proposal_cards:
                try:
                    link = card.find_element(By.CSS_SELECTOR, "a[href*='/proposals/']").get_attribute('href')
                    proposals.append(link)
                except:
                    continue
            
            return proposals
            
        except TimeoutException:
            print("Timeout waiting for proposals to load")
            return []
        
    def scrape_proposal(self, url):
        """Scrape individual proposal details"""
        print(f"Scraping proposal: {url}")
        self.driver.get(url)
        self.human_delay(2, 4)  # Longer delay after page load
        
        try:
            # Wait for content to load
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-test='job-title']")))
            self.human_delay(0.5, 1.5)  # Short delay before scraping
            
            # Extract job details with small delays between operations
            job_data = {
                'title': self._safe_get_text("[data-test='job-title']"),
                'status': self._safe_get_text("[data-test='proposal-status']"),
                'proposal_date': self._safe_get_text("[data-test='proposal-date']"),
                'job_description': self._safe_get_text("[data-test='job-description']"),
                'cover_letter': self._safe_get_text("[data-test='cover-letter']"),
                'client_info': self.get_client_info(),
                'outcome': self.get_job_outcome()
            }
            
            if self.is_successful(job_data['outcome']):
                self.successful_jobs.append(job_data)
            else:
                self.unsuccessful_jobs.append(job_data)
                
        except TimeoutException:
            print(f"Timeout while scraping {url}")
    
    def _safe_get_text(self, selector):
        """Safely get text from an element"""
        try:
            element = self.driver.find_element(By.CSS_SELECTOR, selector)
            return element.text
        except:
            return ""
            
    def get_client_info(self):
        """Extract client information"""
        try:
            return {
                'name': self._safe_get_text("[data-test='client-name']"),
                'rating': self._safe_get_text("[data-test='client-rating']"),
                'location': self._safe_get_text("[data-test='client-location']")
            }
        except:
            return {}
            
    def get_job_outcome(self):
        """Determine if job was successful"""
        try:
            return self._safe_get_text("[data-test='job-outcome']")
        except:
            return "Unknown"
            
    def is_successful(self, outcome):
        """Define what constitutes a successful job"""
        return any(status in outcome.lower() for status in 
                  ['hired', 'completed', 'paid', 'active'])
                  
    def save_results(self):
        """Save scraped data to files"""
        output_dir = Path("data_processing/raw_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save successful jobs
        with open(output_dir / "successful_jobs.json", 'w') as f:
            json.dump(self.successful_jobs, f, indent=2)
            
        # Save unsuccessful jobs
        with open(output_dir / "unsuccessful_jobs.json", 'w') as f:
            json.dump(self.unsuccessful_jobs, f, indent=2)
            
    def run(self):
        """Main execution flow"""
        try:
            print("\nStarting proposal collection...")
            
            # First check active proposals
            print("Checking active proposals...")
            self.driver.get("https://www.upwork.com/nx/find-work/submitted-proposals")
            self.human_delay(2, 4)
            active_proposals = self._get_proposals_from_current_page()
            print(f"Found {len(active_proposals)} active proposals")
            
            # Then check archived proposals
            print("\nChecking archived proposals...")
            self.driver.get("https://www.upwork.com/nx/proposals/archived")
            self.human_delay(2, 4)
            archived_proposals = self._get_proposals_from_current_page()
            print(f"Found {len(archived_proposals)} archived proposals")
            
            # Combine and limit proposals
            all_proposals = active_proposals + archived_proposals
            proposals_to_process = all_proposals[:self.max_proposals]
            print(f"\nProcessing {len(proposals_to_process)} total proposals")
            
            # Process each proposal
            for i, url in enumerate(proposals_to_process, 1):
                print(f"\nProcessing proposal {i} of {len(proposals_to_process)}: {url}")
                self.scrape_proposal(url)
                self.human_delay(1, 2)
            
            # Save results
            self.save_results()
            print("\nScraping completed!")
            print(f"Successful jobs: {len(self.successful_jobs)}")
            print(f"Unsuccessful jobs: {len(self.unsuccessful_jobs)}")
            
        except Exception as e:
            print(f"Error during scraping: {str(e)}")
            raise
        finally:
            print("\nLeaving Chrome session open...")

if __name__ == "__main__":
    scraper = UpworkScraper(max_proposals=5)
    scraper.run() 