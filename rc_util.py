import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import player

service = None
isSetup = False

def setup():
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Set up the Chrome driver
    service = ChromeService(executable_path='chromedriver-win64\\chromedriver.exe')  # Update path to your ChromeDriver
    driver = webdriver.Chrome(service=service, options=chrome_options)
    # Wait for the dropdown menu to be present
    wait = WebDriverWait(driver, 5)  # Increase timeout to desired seconds
    isSetup = True
    return driver, wait

def get_prop_history(url, season_value):
    
    if(not isSetup):
        driver, wait = setup()

    driver.get(url)

    try:
        # Locate the parent section containing the player card
        parent_section = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'section.player-game-log-card')))

        # Scroll the dropdown into view and click using JavaScript
        dropdown_trigger = parent_section.find_element(By.ID, 'player-game-log-season-dropdown')
        driver.execute_script("arguments[0].scrollIntoView(true);", dropdown_trigger)
        driver.execute_script("arguments[0].click();", dropdown_trigger)
        
        # Wait for the dropdown menu options to be visible
        dropdown_menu = wait.until(EC.visibility_of_element_located((By.ID, 'player-game-log-season-dropdown-menu')))
        
        # Select the season
        season_option = dropdown_menu.find_element(By.XPATH, f".//li[@data-value='{season_value}']")
        driver.execute_script("arguments[0].click();", season_option)
        
        # Wait for the table to update (adjust the sleep time if necessary)
        time.sleep(1)
        
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Find the specific section within this parent element
        parent_div = soup.find('section', class_='player-game-log-card')
        
        # Find the unique parent element that contains the table
        table_container = parent_div.find('div', class_='table-overflow--is-scrollable')
        
        driver.quit()

        # Check if the table container was found
        if table_container:
            # Find the specific table within this container
            table = table_container.find('table', class_='table')
            
            # Check if the table was found
            if table:
                # Convert the HTML table to a pandas DataFrame
                df = pd.read_html(str(table))[0]
                
                # Display the DataFrame
                return df[['Date',  'Matchup', 'Prop Line']]
                
                # Optional: Save the DataFrame to a CSV file
                #df.to_csv(f'output_{season_value}.csv', index=False)
            else:
                print(f"Table not found for {season_value} Season")
        else:
            print(f"Table container not found for {season_value} Season")
    except Exception as e:
        print(f"An error occurred: {e}")
        print(driver.page_source)  # Print the page source for debugging


def get_prop_info(url):

    if(not isSetup):
        driver, wait = setup()

    # Fetch the webpage
    driver.get(url)

    try:
        # Locate the parent section containing the offer tabs
        offer_tabs_section = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.offer-tabs__tabs-container')))

        # Find all <a> tags within this section
        a_tags = offer_tabs_section.find_elements(By.TAG_NAME, 'a')

        # Iterate over the first six <a> tags

        output = []
        for a_tag in a_tags[:6]:
            # Extract the href attribute
            #href = a_tag.get_attribute('href')

            # Extract the text within the <span> tags
            soup = BeautifulSoup(a_tag.get_attribute('outerHTML'), 'html.parser')
            spans = soup.find('span', class_='typography')
            prop_info = spans.get_text().split()[0]

            try: 
                prop_info = float(prop_info)
            except Exception as e:
                prop_info = 0


            # Print the href and prop information
            #print(f"Href: {href}")
            output.append(prop_info)
        return output

    except Exception as e:
        print(f"An error occurred: {e}")
        print(driver.page_source)  # Print the page source for debugging
        return 'fucken failed lmao'

def get_upcoming_game(url):

    if(not isSetup):
        driver, wait = setup()
    
    driver.get(url)

    game_url = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'a.participant-info__matchup-link')))
    href = game_url.get_attribute('href')
    return href

def get_schedule(url):
    
    if(not isSetup):
        driver, wait = setup()
    
    driver.get(url)

    #schedule = driver.find_element(By.ID, 'schedule')

     # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()

    # Find the unique parent element that contains the table
    schedule = soup.find('table', class_='stats_table')

    if schedule:
        df = pd.read_html(str(schedule))[0]
    
        # Ensure the 'Date' column is in datetime format (if it's not already)
        df['Date'] = pd.to_datetime(df['Date'], format='%a, %b %d, %Y', errors='coerce')
        
        # Get today's date
        today = datetime.date.today()

        # Filter the DataFrame to only include rows where the 'Date' is today's date
        df = df[df['Date'].dt.date == today]

        # Drop unncessary/empty columns
        df = df.drop(columns=['Arena', 'Notes', 'LOG', 'Attend.']).dropna(how='all', axis='columns')
        
        # Display the filtered DataFrame
        return df

    return 'shit brokey lmfao'

def get_roster(team):

    url = 'https://basketball-reference.com/teams/{0}/2025.html'.format(team)
    if(not isSetup):
        driver, wait = setup()
    
    driver.get(url)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()

    # Find the unique parent element that contains the table
    table = soup.find('table', class_='stats_table')

    if table:
        df = pd.read_html(str(table))[0]

    # Extract the player names and their hrefs
    players = []
    for tr in table.find('tbody').find_all('tr'):
        player_cell = tr.find('td')  # Assuming the player name is in the first cell
        if player_cell and player_cell.find('a'):
            player_name = player_cell.find('a').get_text()
            player_href = player_cell.find('a')['href']
            players.append({'Player': player_name, 'Href': player_href})

    # Create a DataFrame
    df_players = pd.DataFrame(players)
    return df_players

def get_stats(url_path):
    url = f'https://basketball-reference.com/{url_path}'
    if not isSetup:
        driver, wait = setup()

    seasons = []

    current_year = datetime.date.today().year

    end_year = current_year - 5

    for year in range(end_year, current_year):
        driver.get(url.split('.html')[0] + '/gamelog/{0}'.format(year))
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        if soup.find('div', class_='assoc_game_log_summary') is None:
            continue

        # Find the unique parent element that contains the table
        table = soup.find('table', class_='stats_table')

        if table:
            df = pd.read_html(str(table))[0]

        seasons.append(df)

    driver.quit()
    return pd.concat(seasons)
    

#print(get_schedule('https://www.basketball-reference.com/leagues/NBA_2025_games-november.html'))

#print(get_stats('players/h/halibty01.html'))

#print(get_roster('IND'))
# Example usage
#print(get_prop_history('https://www.bettingpros.com/nba/props/dereck-lively-ii/points/', '2023'))
#get_season_data('https://www.bettingpros.com/nba/props/dereck-lively-ii/points/', '2024')
#print(get_upcoming_game('https://www.bettingpros.com/nba/props/tyrese-haliburton/points/'))


