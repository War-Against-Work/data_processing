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
import re

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
                self.jobs_data = []  # Single list for all jobs
                self.max_proposals = max_proposals
                self.proposals_found = 0
                
                # Verify connection by getting current URL
                current_url = self.driver.current_url
                print(f"Connected successfully to: {current_url}")
                
                if "upwork.com" not in current_url:
                    print("Please navigate to Upwork in the debug Chrome window")
                    self.human_delay(2, 3)
                
                # Load existing data if available
                self.load_existing_data()
                
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
            
            # Set a shorter timeout for initial page load check
            short_wait = WebDriverWait(self.driver, 15)  # 15 second timeout
            try:
                # Check for any of these elements to confirm page loaded
                short_wait.until(
                    EC.any_of(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".cover-letter-section")),
                        EC.presence_of_element_located((By.CSS_SELECTOR, "h3.mb-6x.h5")),
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".description.text-body-sm"))
                    )
                )
            except TimeoutException:
                print(f"Page load timeout for {url}")
                return False
            
            # Check for verification
            if self.wait_for_human_verification():
                print("Continuing after verification...")
                self.human_delay(1, 2)
            
            return True
        except Exception as e:
            print(f"Error navigating to {url}: {str(e)}")
            return False

    def wait_for_human_verification(self, timeout=30):
        """Wait for Cloudflare or other verification checks"""
        print("\nWaiting for verification...")
        try:
            # Check for error indicators first
            error_selectors = [
                (By.CSS_SELECTOR, ".air3-card-empty"),  # Empty state
                (By.CSS_SELECTOR, ".error-container"),   # Error state
                (By.CSS_SELECTOR, ".not-found")         # Not found state
            ]
            
            # Quick check for error states
            for by, selector in error_selectors:
                try:
                    element = self.driver.find_element(by, selector)
                    if element.is_displayed():
                        print("Page appears to be empty or invalid")
                        return True  # Allow process to continue
                except:
                    pass

            # Normal verification/content checks
            verification_selectors = [
                (By.ID, "challenge-success-text"),  # Cloudflare success
                (By.CSS_SELECTOR, "[data-qa='card-archived-proposals']"),  # Normal content
                (By.CSS_SELECTOR, ".h5"),  # Job title
                (By.CSS_SELECTOR, ".cover-letter-section"),  # Cover letter
                (By.CSS_SELECTOR, ".description.text-body-sm")  # Job description
            ]
            
            start_time = time.time()
            while time.time() - start_time < timeout:
                for by, selector in verification_selectors:
                    try:
                        element = self.driver.find_element(by, selector)
                        if element.is_displayed():
                            print("Verification completed or page loaded normally")
                            return True
                    except:
                        pass
                time.sleep(1)
            
            print("Verification timeout - continuing anyway")
            return True  # Allow process to continue even on timeout
            
        except Exception as e:
            print(f"Error during verification check: {str(e)}")
            return True  # Allow process to continue on error

    def load_progress(self):
        """Load previously scraped data"""
        try:
            # Load proposal URLs
            if os.path.exists('data_processing/raw_data/proposal_urls.json'):
                with open('data_processing/raw_data/proposal_urls.json', 'r') as f:
                    self.scraped_urls = set(json.load(f))
            else:
                self.scraped_urls = set()
            
            # Load proposal data
            if os.path.exists('data_processing/raw_data/proposals.json'):
                with open('data_processing/raw_data/proposals.json', 'r') as f:
                    self.proposals_data = json.load(f)
            else:
                self.proposals_data = []
            
            print(f"Loaded {len(self.scraped_urls)} previously scraped URLs")
            print(f"Loaded {len(self.proposals_data)} previously scraped proposals")
            
        except Exception as e:
            print(f"Error loading progress: {str(e)}")
            self.scraped_urls = set()
            self.proposals_data = []

    def save_progress(self):
        """Save current progress"""
        try:
            os.makedirs('data_processing/raw_data', exist_ok=True)
            
            # Save URLs
            with open('data_processing/raw_data/proposal_urls.json', 'w') as f:
                json.dump(list(self.scraped_urls), f, indent=2)
            
            # Save proposal data
            with open('data_processing/raw_data/proposals.json', 'w') as f:
                json.dump(self.proposals_data, f, indent=2)
            
            print("\nProgress saved successfully")
            
        except Exception as e:
            print(f"Error saving progress: {str(e)}")

    def get_proposals(self):
        """Navigate to archived proposals page and collect proposal links (max 340)"""
        if not self._reconnect_if_needed():
            raise Exception("Failed to reconnect to Chrome")
        
        try:
            # First try to load URLs from proposal_urls.json
            try:
                with open('data_processing/raw_data/proposal_urls.json', 'r') as f:
                    proposals = json.load(f)
                    print(f"\nLoaded {len(proposals)} proposals from proposal_urls.json")
                    
                    # Process each proposal
                    print("\nScraping proposal details...")
                    for i, proposal in enumerate(proposals, 1):
                        url = proposal['url']
                        initial_outcome = proposal['outcome']
                        print(f"\nProcessing proposal {i} of {len(proposals)}: {url}")
                        print(f"Initial outcome from log: {initial_outcome}")
                        
                        # Skip if already processed
                        if any(job.get('url') == url for job in self.jobs_data):
                            print(f"Skipping already processed proposal: {url}")
                            continue
                        
                        # Get proposal details
                        job_data = self.scrape_proposal(url, initial_outcome)
                        
                        # Take a longer break every 10 proposals
                        if i % 10 == 0:
                            print("Taking a short break...")
                            self.human_delay(5, 8)
                        else:
                            self.human_delay(2, 4)
                    
                    return [p['url'] for p in proposals]
                    
            except FileNotFoundError:
                print("\nNo proposal_urls.json found, collecting URLs from Upwork...")
                self.scraped_urls = set()
                
                # Original URL collection logic
                print("Fetching archived proposals (max 340)...")
                page = 1
                MAX_PROPOSALS = 340
                
                # Load archived proposals page
                print("\nNavigating to archived proposals page...")
                if not self.safe_get("https://www.upwork.com/nx/proposals/archived"):
                    raise Exception("Failed to load archived proposals page")
                
                while len(self.scraped_urls) < MAX_PROPOSALS:
                    print(f"\nProcessing page {page}... ({len(self.scraped_urls)}/{MAX_PROPOSALS} URLs collected)")
                    
                    # Get proposals from current page
                    page_proposals = self._get_proposals_from_current_page()
                    if not page_proposals:
                        print("No proposals found on this page")
                        break
                    
                    # Add new URLs to our set
                    for url in page_proposals:
                        if len(self.scraped_urls) >= MAX_PROPOSALS:
                            break
                        if url not in self.scraped_urls:
                            self.scraped_urls.add(url)
                            # Save progress after each new URL
                            self.save_progress()
                    
                    if len(self.scraped_urls) >= MAX_PROPOSALS:
                        print(f"\nReached maximum of {MAX_PROPOSALS} proposals")
                        break
                    
                    print(f"Total unique URLs found so far: {len(self.scraped_urls)}")
                    
                    # Natural pause between processing pages
                    self.human_delay(2, 4)
                    
                    # Try to go to next page
                    try:
                        next_button = self.driver.find_element(By.CSS_SELECTOR, 
                            "[data-test='next-page']:not([disabled])")
                        
                        if next_button and next_button.is_enabled():
                            print(f"Moving to page {page + 1}...")
                            self.human_delay(1, 2)
                            next_button.click()
                            self.human_delay(3, 5)
                            
                            # Check for verification after pagination
                            if self.wait_for_human_verification():
                                self.human_delay(1, 2)
                            
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
                
                # Process each unprocessed proposal
                unprocessed_urls = [url for url in self.scraped_urls 
                                  if not any(p.get('url') == url for p in self.proposals_data)]
                
                print(f"\nProcessing {len(unprocessed_urls)} unprocessed proposals...")
                
                for i, url in enumerate(unprocessed_urls, 1):
                    print(f"\nProcessing proposal {i} of {len(unprocessed_urls)}: {url}")
                    proposal_data = self.scrape_proposal(url)
                    if proposal_data:
                        self.proposals_data.append(proposal_data)
                        # Save after each successful scrape
                        self.save_progress()
                    
                    # Take a longer break every 10 proposals
                    if i % 10 == 0:
                        print("Taking a short break...")
                        self.human_delay(5, 8)
                    else:
                        self.human_delay(2, 4)
                
                print(f"\nTotal proposals collected: {len(self.proposals_data)}")
                return list(self.scraped_urls)
                
        except Exception as e:
            print(f"Error during proposal collection: {str(e)}")
            raise

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
        
    def _safe_get_text(self, selector, click_more=False):
        """Safely get text from an element, optionally clicking 'more' buttons first"""
        try:
            # If click_more is True, try to click any "more" buttons first
            if click_more:
                try:
                    # Wait a bit for any dynamic content
                    self.human_delay(1, 2)
                    
                    # Find all more buttons
                    more_buttons = self.driver.find_elements(By.CSS_SELECTOR, ".air3-truncation-btn")
                    for button in more_buttons:
                        if button.is_displayed() and "more" in button.text.lower():
                            button.click()
                            self.human_delay(0.5, 1)
                except:
                    pass  # Continue even if clicking more fails
                
            # Now get the text
            element = self.driver.find_element(By.CSS_SELECTOR, selector)
            return element.text
        except:
            return ""

    def get_full_cover_letter(self):
        """Get both cover letter and Q&A content"""
        try:
            # Click all "more" buttons first
            try:
                more_buttons = self.driver.find_elements(By.CSS_SELECTOR, ".air3-truncation-btn")
                for button in more_buttons:
                    if button.is_displayed() and "more" in button.text.lower():
                        button.click()
                        self.human_delay(0.5, 1)
            except:
                pass

            # Get main cover letter
            cover_letter = self._safe_get_text(".cover-letter-section .break.text-pre-line") or ""
            
            # Get Q&A section
            try:
                questions = self.driver.find_elements(By.CSS_SELECTOR, "[data-test='questions-answers'] li")
                qa_text = []
                for q in questions:
                    question = q.find_element(By.CSS_SELECTOR, "strong").text
                    answer = q.find_element(By.CSS_SELECTOR, ".air3-truncation").text
                    qa_text.append(f"{question}\n{answer}")
                
                if qa_text:
                    cover_letter += "\n\nQ&A Section:\n" + "\n\n".join(qa_text)
            except:
                pass
            
            return cover_letter
        except Exception as e:
            print(f"Error getting full cover letter: {str(e)}")
            return ""

    def get_qa_section(self):
        """Get Q&A content as structured data"""
        try:
            # Click all "more" buttons first
            try:
                more_buttons = self.driver.find_elements(By.CSS_SELECTOR, ".air3-truncation-btn")
                for button in more_buttons:
                    if button.is_displayed() and "more" in button.text.lower():
                        button.click()
                        self.human_delay(0.5, 1)
            except:
                pass

            # Get Q&A section
            qa_items = []
            try:
                questions = self.driver.find_elements(By.CSS_SELECTOR, "[data-test='questions-answers'] li")
                for q in questions:
                    question = q.find_element(By.CSS_SELECTOR, "strong").text
                    answer = q.find_element(By.CSS_SELECTOR, ".air3-truncation").text
                    qa_items.append({
                        "question": question,
                        "answer": answer.replace(question, "").strip()  # Remove question from answer text
                    })
            except:
                pass
            
            return qa_items
        except Exception as e:
            print(f"Error getting Q&A section: {str(e)}")
            return []

    def scrape_proposal(self, url, initial_outcome):
        """Scrape individual proposal details with known outcome"""
        print(f"Scraping proposal: {url}")
        
        # Try loading the page up to 3 times
        for attempt in range(3):
            if not self.safe_get(url):
                print(f"Failed to load proposal page (attempt {attempt + 1}/3): {url}")
                if attempt < 2:  # If not last attempt
                    self.human_delay(5, 8)  # Longer delay between retries
                    continue
                else:
                    # If all attempts fail, save minimal data and continue
                    job_data = {
                        'title': "",
                        'status': initial_outcome,
                        'proposal_date': "",
                        'job_description': "",
                        'cover_letter': "",
                        'qa_section': [],
                        'client_info': {},
                        'outcome': initial_outcome,
                        'url': url
                    }
                    self.jobs_data.append(job_data)
                    self.save_results()
                    return job_data
            break  # Page loaded successfully
        
        try:
            # Extract job details, using empty strings for missing fields
            job_data = {
                'title': self._safe_get_text("h3.mb-6x.h5") or "",
                'status': initial_outcome,
                'proposal_date': self._safe_get_text("[data-cy='time-slot']") or "",
                'job_description': self._safe_get_text(".description.text-body-sm", click_more=True) or "",
                'cover_letter': self._safe_get_text(".cover-letter-section .break.text-pre-line", click_more=True) or "",
                'qa_section': self.get_qa_section(),
                'client_info': self.get_client_info() or {},
                'outcome': initial_outcome,
                'url': url
            }
            
            # Save immediately
            self.jobs_data.append(job_data)
            self.save_results()
            
            # Show running totals
            successful = sum(1 for job in self.jobs_data if self.is_successful(job.get('outcome', '')))
            unsuccessful = len(self.jobs_data) - successful
            print(f"\nRunning totals:")
            print(f"Total jobs: {len(self.jobs_data)}")
            print(f"Successful jobs: {successful}")
            print(f"Unsuccessful jobs: {unsuccessful}")
            
            return job_data
            
        except Exception as e:
            print(f"Error scraping proposal: {str(e)}")
            # Save minimal data on error
            job_data = {
                'title': "",
                'status': initial_outcome,
                'proposal_date': "",
                'job_description': "",
                'cover_letter': "",
                'qa_section': [],
                'client_info': {},
                'outcome': initial_outcome,
                'url': url
            }
            self.jobs_data.append(job_data)
            self.save_results()
            return job_data

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
        """Save scraped data to single file"""
        output_dir = Path("data_processing/raw_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all jobs to single file
        with open(output_dir / "jobs_data.json", 'w') as f:
            json.dump(self.jobs_data, f, indent=2)
            
    def run(self):
        """Main execution flow"""
        try:
            print("\nStarting proposal collection...")
            
            # Get and process proposals
            proposals_to_process = self.get_proposals()
            print(f"\nProcessing completed!")
            print(f"Successful jobs: {len(self.jobs_data)}")
            print(f"Unsuccessful jobs: {len(self.jobs_data) - len(self.jobs_data)}")
            
        except Exception as e:
            print(f"Error during scraping: {str(e)}")
            raise
        finally:
            print("\nLeaving Chrome session open...")

    def load_existing_data(self):
        """Load existing jobs data"""
        try:
            # Load all jobs from single file
            if os.path.exists('data_processing/raw_data/jobs_data.json'):
                with open('data_processing/raw_data/jobs_data.json', 'r') as f:
                    self.jobs_data = json.load(f)
            
            # Count successful and unsuccessful for reporting
            successful = sum(1 for job in self.jobs_data if self.is_successful(job.get('outcome', '')))
            unsuccessful = len(self.jobs_data) - successful
            
            print(f"\nLoaded existing data:")
            print(f"Total jobs: {len(self.jobs_data)}")
            print(f"Successful jobs: {successful}")
            print(f"Unsuccessful jobs: {unsuccessful}")
            
        except Exception as e:
            print(f"Error loading existing data: {str(e)}")
            self.jobs_data = []

if __name__ == "__main__":
    scraper = UpworkScraper(max_proposals=5)
    scraper.run() 