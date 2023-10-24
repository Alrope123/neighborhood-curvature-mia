import json
import argparse

def extract_languages(data_path):
    languages = set()

    with open(data_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            language = data.get("meta", {}).get("language")
            if language:
                languages.add(language)

    return languages

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract possible language values from a jsonl file.")
    parser.add_argument("--data_path", type=str, help="Path to the jsonl file")

    args = parser.parse_args()

    result = extract_languages(args.data_path)
    print(f"Possible languages: {', '.join(result)}")

