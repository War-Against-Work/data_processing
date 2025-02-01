from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

def test_connection():
    print("Testing Chrome connection...")
    
    chrome_options = Options()
    chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        print(f"Current URL: {driver.current_url}")
        print(f"Page title: {driver.title}")
        print(f"Window handles: {len(driver.window_handles)}")
        
        print("\nTrying to navigate...")
        driver.get("https://www.upwork.com")
        time.sleep(2)
        print(f"New URL: {driver.current_url}")
        
        return True
    except Exception as e:
        print(f"\nConnection failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_connection() 