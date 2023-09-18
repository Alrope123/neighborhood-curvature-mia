import json
import os
from tqdm import tqdm

def iterate_files(root_dir):
    file_paths = []
    file_names = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
            file_names.append(file_path[len(root_dir):])
    return zip(file_paths, file_names)


def fix_newline_format(text):
    return text.replace('\n', ' ')


if __name__ == '__main__':
    data_dir = "/gscratch/h2lab/alrope/data/redpajama/arxiv/"
    output_dir = "/gscratch/h2lab/alrope/data/redpajama/arxiv_newline_removed/"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i, (data_path, filename) in enumerate(tqdm(iterate_files(data_dir))):
        assert os.path.exists(data_path)
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                dp = json.loads(line)
                dp['text'] = fix_newline_format(dp['text'])
                data.append(fix_newline_format(dp))  
        
        output_path = os.path.join(output_dir, filename)
        output_sub_dir = os.path.dirname(output_path)
        if not os.path.exists(output_sub_dir):
            os.mkdir(output_sub_dir)
        with open(output_path, 'w') as f:
            for dp in data:
                f.write(json.dumps(dp))
                f.write('\n')

    # with open("/gscratch/h2lab/alrope/neighborhood-curvature-mia/debug/out/same_arxiv_document.json", 'r') as f:
    #     data = json.load(f)
    # keys = list(data.keys())
    # selected_key = None
    # for key in keys:
    #     if "Strongly multiplicative linear secret sharing" in key:
    #         selected_key = key
    # text = data[selected_key][0][0]
    # filtered_text = fix_newline_format(text)

    # with open("/gscratch/h2lab/alrope/neighborhood-curvature-mia/debug/out/filtered_text.json", 'w') as f:
    #     json.dump({"filtered_text": filtered_text}, f)
