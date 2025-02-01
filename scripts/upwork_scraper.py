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
    def __init__(self, max_proposals=5, max_retries=3):
        print("Attempting to connect to existing Chrome session...")
        self.max_retries = max_retries
        
        for attempt in range(self.max_retries):
            try:
                chrome_options = Options()
                chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
                chrome_options.add_argument("--disable-notifications")
                
                self.driver = webdriver.Chrome(options=chrome_options)
                self.wait = WebDriverWait(self.driver, 10)
                self.successful_jobs = []
                self.unsuccessful_jobs = []
                self.max_proposals = max_proposals
                self.proposals_found = 0
                
                # Verify connection by getting current URL
                current_url = self.driver.current_url
                print(f"Connected successfully to: {current_url}")
                
                if "upwork.com" not in current_url:
                    print("Please navigate to Upwork in the debug Chrome window")
                    self.human_delay(2, 3)
                
                break  # Connection successful
                
            except Exception as e:
                print(f"\nError on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:  # Last attempt
                    raise Exception("Failed to connect to Chrome after multiple attempts")
                self.human_delay(2, 3)  # Wait before retrying

    def human_delay(self, min_seconds=1, max_seconds=3):
        """Add a random delay to simulate more natural human behavior"""
        # Convert seconds to milliseconds and add random variation
        min_ms = int(min_seconds * 1000)
        max_ms = int(max_seconds * 1000)
        
        # Add some micro-variations to make it more natural
        if min_ms > 100:  # Only add variation for delays longer than 100ms
            min_ms = min_ms - 50
            max_ms = max_ms + 200
        
        delay_ms = random.randint(min_ms, max_ms)
        print(f"Waiting for {delay_ms}ms...")  # Log the actual delay
        time.sleep(delay_ms / 1000)  # Convert back to seconds for sleep
        
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
        
    def _verify_connection(self):
        """Verify that the Chrome connection is still valid"""
        try:
            # Try to access a simple property
            _ = self.driver.current_url
            return True
        except:
            return False

    def _reconnect_if_needed(self):
        """Attempt to reconnect if connection is lost"""
        if not self._verify_connection():
            print("Lost connection to Chrome. Attempting to reconnect...")
            for attempt in range(self.max_retries):
                try:
                    chrome_options = Options()
                    chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
                    chrome_options.add_argument("--disable-notifications")
                    
                    self.driver = webdriver.Chrome(options=chrome_options)
                    self.wait = WebDriverWait(self.driver, 10)
                    
                    # Check for verification after reconnection
                    if self.wait_for_human_verification():
                        print("Verification completed after reconnection")
                    
                    print("Reconnected successfully")
                    return True
                except Exception as e:
                    print(f"Reconnection attempt {attempt + 1} failed: {str(e)}")
                    self.human_delay(2, 3)
            return False
        return True

    def safe_get(self, url):
        """Wrapper for page navigation that handles verification checks"""
        try:
            self.driver.get(url)
            self.human_delay(2, 4)  # Initial load delay
            
            # Check for verification
            if self.wait_for_human_verification():
                print("Continuing after verification...")
                self.human_delay(1, 2)  # Additional delay after verification
                
            return True
        except Exception as e:
            print(f"Error navigating to {url}: {str(e)}")
            return False

    def get_proposals(self):
        """Navigate to archived proposals page and collect all proposal links"""
        if not self._reconnect_if_needed():
            raise Exception("Failed to reconnect to Chrome")
        
        print("Fetching all archived proposals...")
        proposals = []
        page = 1
        
        try:
            # Load archived proposals page
            print("\nNavigating to archived proposals page...")
            if not self.safe_get("https://www.upwork.com/nx/proposals/archived"):
                raise Exception("Failed to load archived proposals page")
            
            while True:  # Continue until we hit the last page
                print(f"\nProcessing page {page}...")
                
                # Get proposals from current page
                page_proposals = self._get_proposals_from_current_page()
                if not page_proposals:
                    print("No proposals found on this page")
                    break
                
                proposals.extend(page_proposals)
                print(f"Total proposals found so far: {len(proposals)}")
                
                # Natural pause between processing pages
                self.human_delay(2, 4)
                
                # Try to go to next page
                try:
                    next_button = self.driver.find_element(By.CSS_SELECTOR, 
                        "[data-test='next-page']:not([disabled])")
                    
                    if next_button and next_button.is_enabled():
                        print(f"Moving to page {page + 1}...")
                        self.human_delay(1, 2)  # Pause before clicking
                        next_button.click()
                        self.human_delay(3, 5)  # Longer pause after page load
                        
                        # Check for verification after pagination
                        if self.wait_for_human_verification():
                            print("Verification completed after pagination")
                            self.human_delay(1, 2)  # Additional pause after verification
                        
                        page += 1
                        
                        # Take a longer break every 5 pages
                        if page % 5 == 0:
                            print("Taking a short break...")
                            self.human_delay(5, 8)
                    else:
                        print("Reached the last page")
                        break
                except Exception as e:
                    print(f"Error navigating to next page: {str(e)}")
                    break
            
            print(f"\nTotal proposals collected: {len(proposals)}")
            
            # Process each proposal with natural pacing
            for i, url in enumerate(proposals, 1):
                print(f"\nProcessing proposal {i} of {len(proposals)}: {url}")
                self.scrape_proposal(url)
                
                # Take a longer break every 10 proposals
                if i % 10 == 0:
                    print("Taking a short break...")
                    self.human_delay(5, 8)
                else:
                    self.human_delay(2, 4)  # Normal pause between proposals
            
            # Save after processing
            self.save_results()
            print("\nSaved results to:")
            print("- data_processing/raw_data/successful_jobs.json")
            print("- data_processing/raw_data/unsuccessful_jobs.json")
            
            return proposals
            
        except Exception as e:
            print(f"Error during scraping: {str(e)}")
            raise
        finally:
            print("\nLeaving Chrome session open...")
    
    def wait_for_human_verification(self):
        """Wait for user to complete Cloudflare or other human verification"""
        try:
            # Common Cloudflare challenge indicators
            cloudflare_selectors = [
                "#challenge-running",
                "#challenge-stage",
                "[data-translate='error.client_error']",
                "iframe[title*='challenge']",
                "#cf-challenge-running"
            ]
            
            # Check if any Cloudflare elements are present
            for selector in cloudflare_selectors:
                try:
                    if self.driver.find_element(By.CSS_SELECTOR, selector):
                        print("\nHuman verification detected!")
                        print("Please complete the verification in the browser...")
                        input("Press Enter once you've completed the verification...")
                        self.human_delay(2, 3)  # Give page time to load after verification
                        return True
                except:
                    continue
                
            return False
            
        except Exception as e:
            print(f"Error checking for verification: {str(e)}")
            return False

    def _get_proposals_from_current_page(self):
        """Extract proposal links from current page"""
        try:
            # Check for verification before proceeding
            if self.wait_for_human_verification():
                print("Continuing after verification...")
            
            # Wait for the archived proposals section to load
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-qa='card-archived-proposals']")))
            
            # Find the archived proposals section specifically
            archived_proposals_section = self.driver.find_element(By.CSS_SELECTOR, "[data-qa='card-archived-proposals']")
            
            # Find all proposal rows within the archived proposals section
            proposal_cards = archived_proposals_section.find_elements(By.CSS_SELECTOR, "[data-qa^='item']")
            
            print(f"Found {len(proposal_cards)} archived proposal cards on current page")
            
            # Extract links from cards
            proposals = []
            for card in proposal_cards:
                try:
                    # Find link using the specific job info cell
                    link_element = card.find_element(By.CSS_SELECTOR, "td[data-cy='job-info'] a")
                    if link_element:
                        link = link_element.get_attribute('href')
                        
                        # Get status for logging
                        try:
                            status = card.find_element(By.CSS_SELECTOR, "[data-qa='reason-slot']").text
                            print(f"Found proposal link: {link} (Status: {status})")
                        except:
                            print(f"Found proposal link: {link}")
                        
                        proposals.append(link)
                    else:
                        print("No link found in card")
                except Exception as e:
                    print(f"Error extracting link from card: {str(e)}")
                    continue
            
            return proposals
            
        except TimeoutException:
            print("Timeout waiting for archived proposals section to load")
            return []
        except Exception as e:
            print(f"Error getting proposals from page: {str(e)}")
            return []
        
    def scrape_proposal(self, url):
        """Scrape individual proposal details"""
        print(f"Scraping proposal: {url}")
        if not self.safe_get(url):
            print(f"Failed to load proposal page: {url}")
            return
        
        try:
            # Wait for content to load using the job title selector
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".h5")))
            self.human_delay(0.5, 1.5)
            
            # Click "more" button if it exists to expand content
            try:
                more_button = self.driver.find_element(By.CSS_SELECTOR, ".air3-truncation-btn")
                if more_button:
                    more_button.click()
                    self.human_delay(1, 2)
                    
                    # Check for verification after expanding content
                    if self.wait_for_human_verification():
                        print("Verification completed after expanding content")
            except:
                pass
            
            # Extract job details
            job_data = {
                'title': self._safe_get_text("h3.mb-6x.h5"),
                'status': self.get_job_outcome(),
                'proposal_date': self._safe_get_text("[data-cy='time-slot']"),
                'job_description': self._safe_get_text(".description.text-body-sm"),
                'cover_letter': self._safe_get_text(".cover-letter-section .break.text-pre-line"),
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
            # Wait for client info section to load
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-test='about-client-container']")))
            
            client_info = {
                'location': self._safe_get_text("[data-qa='client-location'] strong"),
                'city': self._safe_get_text("[data-qa='client-location'] span.nowrap:first-child"),
                'jobs_posted': self._safe_get_text("[data-qa='client-job-posting-stats'] strong"),
                'hire_info': self._safe_get_text("[data-qa='client-job-posting-stats'] div"),
                'total_spent': self._safe_get_text("[data-qa='client-spend'] span span"),
                'hires_info': self._safe_get_text("[data-qa='client-hires']"),
                'avg_hourly_rate': self._safe_get_text("[data-qa='client-hourly-rate']"),
                'hours': self._safe_get_text("[data-qa='client-hours']"),
                'company_size': self._safe_get_text("[data-qa='client-company-profile-size']"),
                'member_since': self._safe_get_text("[data-qa='client-contract-date'] small"),
                'rating': self._safe_get_text("[data-testid='buyer-rating'] .air3-rating-value-text"),
                'reviews': self._safe_get_text("[data-testid='buyer-rating'] span.nowrap"),
                'payment_verified': self._safe_get_text(".text-light-on-muted.text-caption")
            }
            
            # Clean up any empty values
            return {k: v.strip() for k, v in client_info.items() if v.strip()}
            
        except Exception as e:
            print(f"Error getting client info: {str(e)}")
            return {}
            
    def get_job_outcome(self):
        """Determine the outcome/status of the proposal"""
        try:
            # Check for various possible status indicators
            status_selectors = [
                "[data-test='withdraw-reason'] span",  # For withdrawn proposals
                "[data-test='proposal-status']",       # For general status
                "[data-test='job-status']",           # For job status
                ".mb-6x span"                         # Alternative location
            ]
            
            for selector in status_selectors:
                status = self._safe_get_text(selector)
                if status:
                    print(f"Found status: {status}")
                    return status.strip()
            
            return "Unknown"
        except Exception as e:
            print(f"Error getting job outcome: {str(e)}")
            return "Unknown"
            
    def is_successful(self, outcome):
        """Define what constitutes a successful job"""
        successful_statuses = [
            'hired',
            'completed', 
            'paid', 
            'active',
            'in progress'
        ]
        return any(status in outcome.lower() for status in successful_statuses)
                  
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
            
            # Get and process proposals
            proposals_to_process = self.get_proposals()
            print(f"\nProcessing completed!")
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