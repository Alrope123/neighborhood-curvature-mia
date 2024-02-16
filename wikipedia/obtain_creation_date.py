import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import argparse
import pickle as pkl
import os
from tqdm import tqdm
import random

def downsample_set(input_set, factor):
    """
    Downsamples a set by a given factor using a random seed.

    Parameters:
        input_set (set): The set to be downsampled.
        factor (float): The factor by which to downsample the set. Should be between 0 and 1.

    Returns:
        set: The downsampled set.
    """
    
    output_set = set()

    for element in input_set:
        if random.random() < factor:
            output_set.add(element)
    
    return output_set


def get_wikipedia_creation_timestamp(article_title):
    # URL encoding the article title to properly format the URL
    encoded_title = quote(article_title)

    # Create URL for the Wikipedia information page of the article
    info_url = f"https://en.wikipedia.org/w/index.php?title={encoded_title}&action=info"
    
    # Fetch the page content
    response = requests.get(info_url)
    if response.status_code != 200:
        print("Failed to fetch the webpage at {} for {}.".format(info_url, article_title))
        return None
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the table containing article information
    row = soup.find('tr', {'id': 'mw-pageinfo-firsttime'})
    if not row:
        print("Failed to find the page creation time row for {}.".format(article_title))
        return None
    
    # Loop through the table rows
    timestamp_cell = row.find_all('td')
    if timestamp_cell:
        return timestamp_cell[-1].text.strip()
    
    print("Failed to find the creation timestamp for {}.".format(article_title))
    return None



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_path', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/wikipedia/out")
    parser.add_argument('--perfect_set_path', type=str, default=None)
    parser.add_argument('--set_name', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/wikipedia/out")
    parser.add_argument('--save_interval', type=int, default=10000)
    parser.add_argument('--downsample_factor', type=float, default=1.0)

    args = parser.parse_args()

    new_name = args.set_name.replace('.pkl', '_w_time.pkl')
    out_path = os.path.join(args.out_dir, new_name)
    print("Writing to {}".format(out_path))

    # load the set of title
    with open(os.path.join(args.set_path, args.set_name), 'rb') as f:
        title_set = pkl.load(f)
    if args.perfect_set_path:
        with open(os.path.join(args.perfect_set_path, "perfect_set.pkl"), 'rb') as f:
            title_set = pkl.load(f)
    else:
        with open(os.path.join(args.set_path, args.set_name), 'rb') as f:
            title_set = pkl.load(f)
    random.seed(2023)
    print("Size of the set: {}".format(len(title_set)))
    if args.downsample_factor < 1.0:
        title_set = downsample_set(title_set, args.downsample_factor)
        print("Size of the set after downsampling: {}".format(len(title_set)))

    # Initiate a dictionary
    if os.path.exists(out_path):
        with open(out_path, 'rb') as f:
            article_to_timestamp = pkl.load(f)
    else:
        article_to_timestamp = {}

    print("Current dictionary size is {}.".format(len(article_to_timestamp)))
    title_set = set([title for title in list(title_set) if title not in article_to_timestamp])
    print("Size of the set need to obtain creation date: {}".format(len(title_set)))
    
    for i, title in enumerate(tqdm(title_set)):
        if title not in article_to_timestamp:
            article_to_timestamp[title] = get_wikipedia_creation_timestamp(title)

        if i % args.save_interval == 0:
            with open(out_path, 'wb') as f:
                pkl.dump(article_to_timestamp, f)

    with open(out_path, 'wb') as f:
        pkl.dump(article_to_timestamp, f)

