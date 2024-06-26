import os
import json
import csv
import numpy as np

if __name__ == "__main__":
    metadata_path = "/gscratch/h2lab/sewon/data/books3/metadata/metadata.jsonl"
    titles = set()
    authors = set()
    author_to_titles = {}

    with open(metadata_path, 'r') as f:
        for line in f:
            dp = json.loads(line)
            title = dp['file'].split('/')[-1].split('.')[0].split('-')[0].strip()
            author = dp['author'].strip() 
            titles.add(title)
            authors.add(author)
            if author not in author_to_titles:
                author_to_titles[author] = []
            author_to_titles[author].append(title)

    print("There are {} authors and {} titles".format(len(authors), len(titles)))
    print("On average, each author has {} books".format(np.mean([len(titles) for _, titles in author_to_titles.items()])))
    print("Number of author that has at least 5 books: {}".format(len([author for author, titles in author_to_titles.items() if len(titles) > 5])))

    # Define the path to your TSV file
    tsv_file_path = '/gscratch/h2lab/alrope/neighborhood-curvature-mia/book/gpt4-books.tsv'
    # Initialize an empty list to store the extracted data
    book_list = []
    # Open the TSV file for reading
    with open(tsv_file_path, 'r', newline='', encoding='utf-8') as tsv_file:
        # Create a CSV reader object with tab as the delimiter
        tsv_reader = csv.DictReader(tsv_file, delimiter='\t')
        # Iterate through each row in the TSV file
        for row in tsv_reader:
            # Extract the desired columns
            author = row['Author']
            title = row['Title']
            year = row['Year']
            # Create a dictionary for the current row and append it to the data_list
            data_dict = {'Author': author, 'Title': title, 'Year': year}
            book_list.append(data_dict)
    
    for key in ['Author', 'Title']:
        books_not_copyrighted = []
        books_copyrighted = []
        for entry in book_list:
            if entry['Year'] <= "1928":
                books_not_copyrighted.append(entry[key] in titles if key == "Title" else entry[key] in authors)
            else:
                books_copyrighted.append(entry[key] in titles if key == "Title" else entry[key] in authors)
        print("In the {} list:".format(key))
        print("\tFor not copyrighted books, the percentage of books in Books3 is {}".format(sum(books_not_copyrighted) / len(books_not_copyrighted)))
        print("\tFor copyrighted books, the percentage of books in Books3 is {}".format(sum(books_copyrighted) / len(books_copyrighted)))

    

    