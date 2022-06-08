"""
prepare NIKL dataset for pretraining
=> json to txt file(libe by line)
"""

import os
import re
import json
import argparse
from kss import split_sentences
from time import sleep
from tqdm import tqdm

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class Processor:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.save_dir = args.save_dir
        self.do_rearrange = args.do_rearrange
        self.category = 'utterance' if args.do_rearrange else 'paragraph'
        print(f'processing {self.data_dir}...')


    def write(self, file_name, sentences):
        assert file_name.endswith('.json')
        file_name = file_name.split('.json')[0] + '.txt'
        with open(os.path.join(self.save_dir, file_name), 'w+') as f:
            f.write('\n'.join(sentences))


    def process(self):
        def get_file_list():
            file_list = os.listdir(self.data_dir)
            json_files = [f for f in file_list if f.endswith("json")]
            return json_files
        

        json_files = get_file_list()
        for json_file_name in tqdm(json_files, desc="processing json files"):
            json_file = os.path.join(self.data_dir, json_file_name)
            txt_file_name = json_file_name.split('.json')[0] + '.txt'
            if os.path.exists(os.path.join(self.save_dir, txt_file_name)):
                continue
            sentences = list()
            with open(json_file) as f:
                data = json.load(f)
            doc = data['document']
            string = ""
            for article in doc:
                for l in article[self.category]:
                    text = l['form'].replace('\n', '').replace('\r', '')
                    text = re.sub('[^ㄱ-ㅎ가-힣ㅏ-ㅣa-zA-Z0-9\.\/\!\@\#\$\%\^\&?\;\:\`\'\-\,\|\~\_\(\)\[\]\{\}\"\s\+-=\\\\]', '', text)
                    sentences.append(text)
            """    
            # for only KParlty
            for l in doc[self.category]:
                sentences.append(l['form'].replace('\n','').replace('\r',''))
            """
            self.write(json_file_name, sentences)
        



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="dataset directory")
    parser.add_argument("--save-dir", required=True, help="directory for save the results")
    parser.add_argument("--do-rearrange", action='store_true', help="whether to merge & split(kss) forms")

    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    processor = Processor(args)
    processor.process()



if __name__ == "__main__":
    main()
