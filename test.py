import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Set up the Chrome driver
service = ChromeService(executable_path="C:\\Users\\Abraham\\Documents\\C964\\chromedriver-win64\\chromedriver.exe")  # Update path to your ChromeDriver
driver = webdriver.Chrome(service=service, options=chrome_options)

# Fetch the webpage
url = 'https://www.bettingpros.com/nba/props/tyrese-haliburton/points/'  # Replace with your URL
driver.get(url)

# Wait for the page to load content dynamically
time.sleep(5)  # Adjust the sleep time as needed to ensure the page loads completely

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(driver.page_source, 'html.parser')

parent_div = soup.find('section', class_='player-game-log-card')

# Find the specific table with the class 'table'
table = parent_div.find('table', class_='table')

# Check if the table was found
if table:
    # Convert the HTML table to a pandas DataFrame
    df = pd.read_html(str(table))[0]

    # Display the DataFrame
    print(df)

    # Optional: Save the DataFrame to a CSV file
    df.to_csv('output.csv', index=False)
else:
    print("Table not found")

# Close the browser
driver.quit()
