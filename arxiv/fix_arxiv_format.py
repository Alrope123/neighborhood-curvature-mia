import json

def fix_newline_format(text):
    return text.replace('\n', '')

if __name__ == '__main__':
    with open("/gscratch/h2lab/alrope/neighborhood-curvature-mia/debug/out/same_arxiv_document.json", 'r') as f:
        data = json.load(f)
    keys = list(data.keys())
    selected_key = None
    for key in keys:
        if "Strongly multiplicative linear secret sharing" in key:
            selected_key = key
    text = data[selected_key][0][0]
    filtered_text = fix_newline_format(text)

    with open("/gscratch/h2lab/alrope/neighborhood-curvature-mia/debug/out/filtered_text.json", 'w') as f:
        json.dump({"filtered_text": filtered_text}, f)
