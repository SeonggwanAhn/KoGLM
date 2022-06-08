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
        entire_string = list()  # for spoken
        file_name_list = list()  # for spoken
        for json_file_name in tqdm(json_files, desc="processing json files"):
            json_file = os.path.join(self.data_dir, json_file_name)
            txt_file_name = json_file_name.split('.json')[0] + '.txt'
            if os.path.exists(os.path.join(self.save_dir, txt_file_name)):
                continue
            sentences = list()
            with open(json_file) as f:
                data = json.load(f)
            doc = data['document']
            """
            if self.category == 'utterance':
                string = ""
                for article in doc:
                    for l in article[self.category]:
                        text = l['form'].replace('\n', '').replace('\r', '')
                        text = re.sub('[^ㄱ-ㅎ가-힣ㅏ-ㅣa-zA-Z0-9\.\/\!\@\#\$\%\^\&?\;\:\`\'\-\,\|\~\_\(\)\[\]\{\}\"\s\+-=\\\\]', '', text)
                        string += text + " "
                entire_string.append(string.strip())
                file_name_list.append(json_file_name)
            """
            if self.category == 'paragraph' or self.category == 'utterance':
                """
                for article in doc:
                    for l in article[self.category]:
                        text = l['form'].replace('\n', '').replace('\r', '')
                        text = re.sub('[^ㄱ-ㅎ가-힣ㅏ-ㅣa-zA-Z0-9\.\/\!\@\#\$\%\^\&?\;\:\`\'\-\,\|\~\_\(\)\[\]\{\}\"\s\+-=\\\\]', '', text)
                        if len(text) == 0:
                            continue
                        sentences.append(text)
                """
                # for only KParlty
                for l in doc[self.category]:
                    text = l['form'].replace('\n', '').replace('\r', '')
                    text = re.sub('[^ㄱ-ㅎ가-힣ㅏ-ㅣa-zA-Z0-9\.\/\!\@\#\$\%\^\&?\;\:\`\'\-\,\|\~\_\(\)\[\]\{\}\"\s\+-=\\\\]', '', text)
                    if len(text) == 0:
                        continue
                    sentences.append(text)
                
                self.write(json_file_name, sentences)
        
        """
        batch_size = 1
        if entire_string != [] and file_name_list != []:
            assert len(entire_string) == len(file_name_list)
            for str_batch, file_batch in zip(batch(entire_string, batch_size), batch(file_name_list, batch_size)):
                sentence_list = split_sentences(str_batch, backend="mecab")
                # print(f'str_batch len - {len(str_batch)}')
                # print(f'sentence_list len - {len(sentence_list)}')
                # print(sentence_list, str_batch)
                print(file_batch)
                reduce_size = 10
                cnt = 0
                while len(sentence_list) != len(file_batch):
                    batch_reduce = list()
                    for text in str_batch:
                        batch_reduce.append(text[:reduce_size *-1])
                    sentence_list = split_sentences(batch_reduce, backend="mecab")
                    if cnt < 15:
                        reduce_size += 10
                    elif cnt < 30:
                        reduce_size += 30
                    elif cnt < 45:
                        reduce_size += 50
                    else:
                        reduce_size += 100
                    cnt += 1
                    print(f'batch_reduce - {batch_reduce}')
                assert len(sentence_list) == len(file_batch)
                for sent, file_name in zip(tqdm(sentence_list, desc='writing...'), file_batch):
                    self.write(file_name, sent)
        """    
            # bug => not same list size with entire_string
            # sentence_list = split_sentences(entire_string, backend="mecab")
            # for sent, file_name in zip(tqdm(sentence_list, desc='writing...'), file_name_list):
            #     self.write(file_name, sent)



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
