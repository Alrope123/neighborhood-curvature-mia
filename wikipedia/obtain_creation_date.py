import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

def get_wikipedia_creation_timestamp(article_title):
    # URL encoding the article title to properly format the URL
    encoded_title = quote(article_title)

    # Create URL for the Wikipedia information page of the article
    info_url = f"https://en.wikipedia.org/w/index.php?title={encoded_title}&action=info"
    
    # Fetch the page content
    response = requests.get(info_url)
    if response.status_code != 200:
        print("Failed to fetch the webpage at {}.".format(info_url))
        return None
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the table containing article information
    row = soup.find('tr', {'id': 'mw-pageinfo-firsttime'})
    if not row:
        print("Failed to find the page creation time row.")
        return None
    
    # Loop through the table rows
    timestamp_cell = row.find_all('td')
    if timestamp_cell:
        return timestamp_cell[-1].text.strip()
    
    print("Failed to find the creation timestamp.")
    return None

if __name__ == '__main__':
    article_title = "Barack Obama"
    import time
    n_iterations = 100
    startTime = time.time()
    for i in range(n_iterations):
        timestamp = get_wikipedia_creation_timestamp(article_title)
    # if timestamp:
    #     print(f"The article '{article_title}' was created on: {timestamp}")
    endTime = time.time()
    time_taken = endTime - startTime
    print("Used {} to process {} articles. Average rate: {} articles/s".format(time_taken, n_iterations, n_iterations/time_taken))
