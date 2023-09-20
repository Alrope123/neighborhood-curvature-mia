import os
import json
import csv

if __name__ == "__main__":
    metadata_path = "/gscratch/h2lab/sewon/data/books3/metadata/metadata.jsonl"
    titles = set()
    authors = set()

    with open(metadata_path, 'r') as f:
        for line in f:
            dp = json.loads(line)
            titles.add(dp['file'].split('\/')[-1].split['.'][0].split['-'][0].strip())
            authors.add(dp['author'].strip())
    
    print(titles.pop())


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
        print("In the author list:")
        print("\tFor not copyrighted books, the percentage of books in Books3 is {}".format(sum(books_not_copyrighted) / len(books_not_copyrighted)))
        print("\tFor copyrighted books, the percentage of books in Books3 is {}".format(sum(books_copyrighted) / len(books_copyrighted)))

    

    