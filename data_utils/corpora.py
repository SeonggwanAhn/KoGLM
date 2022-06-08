# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""several datasets with preset arguments"""
from .datasets import json_dataset, csv_dataset
import os
import json
import random
import tqdm
from multiprocessing import Queue, Process
from queue import Empty
from collections import defaultdict
from torch.utils import data
from .lazy_loader import LazyLoader
from utils import print_rank_0
# from kss import split_sentences
import multiprocessing, logging

NUM_PROCESSES = 8  # 100


def punctuation_standardization(string: str):
    punctuation_dict = {"\u201c": "\"", "\u201d": "\"", "\u2019": "'", "\u2018": "'", "\u2013": "-"}
    for key, value in punctuation_dict.items():
        string = string.replace(key, value)
    return string


class KeyDataset(data.Dataset):
    def __init__(self, text_loader, mask_loader, **kwargs):
        self.texts = text_loader
        self.masks = mask_loader
        self.is_lazy = False
        if isinstance(self.texts, LazyLoader) and isinstance(self.masks, LazyLoader):
            self.text_lens = self.texts.lens
            self.is_lazy = True

    def get_text_len(self, idx):
        return self.text_lens[idx]

    def __getitem__(self, index):
        text = self.texts[index]
        mask_length = self.masks[index]
        mask = []
        for i, length in enumerate(mask_length):
            if i % 2 == 0:
                mask += [0] * length
            else:
                mask += [1] * length
        assert len(text) == len(mask)
        return {"tokens": text, "loss_masks": mask}

    def __len__(self):
        return len(self.texts)


class PromptDataset(data.Dataset):
    def __init__(self, prompt_loader, text_loader, tokenizer=None, to_tokenize=False, **kwargs):
        self.prompts = prompt_loader
        self.texts = text_loader
        self.tokenizer = tokenizer
        self.to_tokenize = to_tokenize
        if isinstance(self.prompts, LazyLoader) and isinstance(self.texts, LazyLoader):
            self.prompt_lens = self.prompts.lens
            self.text_lens = self.texts.lens
            self.is_lazy = True

    def get_text_len(self, idx):
        return self.prompt_lens[idx] + self.text_lens[idx]

    def __getitem__(self, index):
        prompt = self.prompts[index]
        text = self.texts[index]
        if self.to_tokenize:
            prompt = self.tokenizer.EncodeAsIds(prompt).tokenization
            text = self.tokenizer.EncodeAsIds(text).tokenization
        return {"tokens": prompt + text, "loss_masks": [0] * len(prompt) + [1] * len(text)}

    def __len__(self):
        return len(self.prompts)


class DataReader:
    PATH = None
    assert_str = None
    reserve_punct = False
    split_row = True
    TASK_QUEUE_LIMIT = 10000000
    DONE_QUEUE_LIMIT = 10000000

    def tokenize_worker(self, input, output, info, tokenizer, tokenize):
        raise NotImplementedError

    def print_info(self, info):
        pass

    def __init__(self, writers, tokenizer=None, tokenize=False, **kwargs):
        assert os.path.exists(self.PATH), self.assert_str
        print_rank_0(f"Creating dataset from {self.PATH}")
        self.tokenizer = tokenizer
        self.tokenize = tokenize
        self.writers = writers

    def process(self):
        # logger = multiprocessing.log_to_stderr()
        # logger.setLevel(multiprocessing.SUBDEBUG)

        if os.path.isdir(self.PATH):
            paths = [os.path.join(top, name) for top, _, names in os.walk(self.PATH) for name in names]
            # paths = [entry.path for entry in os.scandir(self.PATH) if
            #          not entry.is_dir() and not entry.name.endswith("bz2")]
        else:
            paths = [self.PATH]
        task_queue, done_queue, info_queue = Queue(maxsize=self.TASK_QUEUE_LIMIT), Queue(
            maxsize=self.DONE_QUEUE_LIMIT), Queue()
        processes = []
        for i in range(NUM_PROCESSES):
            process = Process(target=self.tokenize_worker,
                              args=(task_queue, done_queue, info_queue, self.tokenizer, self.tokenize))
            process.start()
            processes.append(process)

        def read_input_to_queue():
            for path in paths:
                print_rank_0(f"Start reading {path}")
                with open(path) as file:
                    if self.split_row:
                        for row in file:
                            task_queue.put(row)
                    else:
                        items = json.load(file)
                        if "NIKL" in path:
                            for item in items['document']:
                                # why ??
                                if item in ['id', 'metadata']:
                                    continue
                                # for paragraph in item[self.category]:
                                #     task_queue.put(paragraph)
                                # origin
                                task_queue.put(item)
                        else:
                            for item in items["RECORDS"]:
                                task_queue.put(item)
            print_rank_0("Read input complete")
            for i in range(len(processes)):
                task_queue.put('STOP')

        process = Process(target=read_input_to_queue)
        process.start()
        count = len(processes)
        progress_bar = tqdm.tqdm()
        while True:
            data = done_queue.get()
            if data == 'COMPLETE':
                count -= 1
                if count == 0:
                    break
            else:
                self.write_result(data, self.writers)
                progress_bar.update()
        progress_bar.close()
        self.print_info(info_queue)

    @staticmethod
    def write_result(data, writers):
        raise NotImplementedError

    @staticmethod
    def get_token_count(contents):
        return sum(map(len, contents))

    @classmethod
    def process_sample(cls, text, tokenizer, tokenize):
        if isinstance(text, str) and tokenize:
            if not cls.reserve_punct:
                text = punctuation_standardization(text)

            text = tokenizer.EncodeAsIds(text).tokenization if text else []
            # text = tokenizer.EncodeAsIds(text) if text else []
            # print(f'\ntext.tokenization : {text}\n\n')
        return text

    @staticmethod
    def trim_field(content, max_length):
        if len(content) > max_length:
            content = content[:max_length]
            content += "......"
        return content

    def process_line(self, data, tokenizer, tokenize):
        raise NotImplementedError


class PromptReader(DataReader):
    is_json = True

    def tokenize_worker(self, input, output, info, tokenizer, tokenize):
        # what is this iterator??
        for row in iter(input.get, 'STOP'):
            if row:
                # print(f'row in tokenize_worker : {row}')
                if self.is_json:
                    row = row.rstrip()
                    row = json.loads(row)
                prompts, texts = self.process_line(row, tokenizer, tokenize)
                for prompt, text in zip(prompts, texts):
                    output.put((prompt, text))
        output.put("COMPLETE")

    @staticmethod
    def write_result(data, writers):
        prompt, text = data
        writers['prompt'].write(prompt)
        writers['text'].write(text)


class KeyReader(DataReader):
    PATH = '/root/data/wikipedia/wiki-key.txt'
    assert_str = "make sure to set PATH for wikipedia data_utils/corpora.py"

    def process_line(self, data, tokenizer, tokenize):
        keys, contents = data['key'], data["content"]
        assert len(keys) == len(contents)
        for i in range(1, len(keys)):
            keys[i] = " " + keys[i]
        contents = [" " + content for content in contents]
        keys = [tokenizer.EncodeAsIds(key).tokenization for key in keys]
        contents = [tokenizer.EncodeAsIds(content).tokenization for content in contents]
        summary = sum(keys, [])
        summary_prefix = self.process_sample("Summary: ", tokenizer, tokenize)
        summary_mask = [len(summary_prefix), len(summary)]
        summary = summary_prefix + summary
        text, text_mask = [], []
        for key, content in zip(keys, contents):
            content = content + [tokenizer.get_command('eop').Id]
            text += key
            text += content
            text_mask.append(len(key))
            text_mask.append(len(content))
        return (summary, summary_mask), (text, text_mask)

    def tokenize_worker(self, input, output, info, tokenizer, tokenize):
        for row in iter(input.get, 'STOP'):
            data = json.loads(row)
            summary, content = self.process_line(data, tokenizer, tokenize)
            output.put((summary, content))
        output.put("COMPLETE")

    @staticmethod
    def write_result(data, writers):
        summary, content = data
        writers['text'].write(summary[0])
        writers['mask'].write(summary[1])
        writers['text'].write(content[0])
        writers['mask'].write(content[1])


class zhihu(PromptReader):
    PATH = "/dataset/fd5061f6/data/tokenize_data/zhihu.lazy"
    reserve_punct = True
    assert_str = "make sure to set PATH for zhihu data_utils/corpora.py"
    qtitle_prefix = "问题："
    qcontent_prefix = "问题描述："
    user_prefix = "回答用户："
    answer_prefix = " 回答："

    # qtitle_prefix = []
    # qcontent_prefix = []
    # user_prefix = []
    # answer_prefix = []

    def process_line(self, data, tokenizer, tokenize):
        prompts, texts = [], []
        ans_length = len(data.get("ans-content", ""))
        ans_up = data.get("ans-up-num", "")
        ans_up = int(ans_up) if ans_up else 0
        if ans_length > 100 or ans_up > 1000:
            qtitle = data["q_title"]
            qcontent = data["q-content"]
            if qcontent is None:
                qcontent = ""
            qcontent = self.trim_field(qcontent, max_length=100)
            user = data.get("user-signature", "")
            prompt = self.qtitle_prefix + qtitle + self.qcontent_prefix + qcontent + self.user_prefix + user + self.answer_prefix
            text = data["ans-content"]
            prompt, text = self.process_sample(prompt, tokenizer, tokenize), self.process_sample(text, tokenizer,
                                                                                                 tokenize)
            prompts.append(prompt)
            texts.append(text)
        # prompt = data["q_title"] + data["q-content"] + data["user-signature"]
        # text = data["ans-content"]
        # prompts.append(prompt)
        # texts.append(text)
        return prompts, texts


class zhidao(PromptReader):
    PATH = "/root/data/zhidao/zhidao"
    reserve_punct = True
    assert_str = "make sure to set PATH for zhidao data_utils/corpora.py"
    qtitle_prefix = "问题："
    qcontent_prefix = "问题描述："
    answer_prefix = "回答："

    def process_line(self, data, tokenizer, tokenize):
        if "title" not in data:
            return [], []
        prompts, texts = [], []
        qtitle = data["title"]
        qcontent = data.get("content", "")
        qcontent = self.trim_field(qcontent, max_length=100)
        prompt = self.qtitle_prefix + qtitle + self.qcontent_prefix + qcontent + self.answer_prefix
        prompt = self.process_sample(prompt, tokenizer, tokenize)
        if "best_answer" in data:
            text = data["best_answer"]["content"]
            if len(text) > 10:
                text = self.process_sample(text, tokenizer, tokenize)
                prompts.append(prompt)
                texts.append(text)
        for answer in data.get("other_answers", []):
            text = answer["content"]
            if len(text) > 100:
                text = self.process_sample(text, tokenizer, tokenize)
                prompts.append(prompt)
                texts.append(text)
        return prompts, texts


class baike(PromptReader):
    PATH = "/dataset/fd5061f6/data/tokenize_data/baike.lazy"
    reserve_punct = True
    assert_str = "make sure to set PATH for baike data_utils/corpora.py"

    def process_line(self, data, tokenizer, tokenize):
        prompts, texts = [], []
        text = data.get("title", "") + data.get("abstract", "") + data.get("content", "")
        if text:
            p, t = self.process_sample("", tokenizer, tokenize), self.process_sample(text, tokenizer, tokenize)
            prompts.append(p)
            texts.append(t)
        return prompts, texts


class wikipedia(PromptReader):
    """
    dataset for wikipedia with arguments configured for convenience

    command line usage: `--train-data wikipedia`
    """
    # PATH = '/dataset/data/wiki.txt'
    PATH = '/root/data/bert_data/wiki.txt'
    assert_str = "make sure to set PATH for wikipedia data_utils/corpora.py"

    def process_line(self, data, tokenizer, tokenize):
        text = data['text']
        prompt, text = self.process_sample("", tokenizer, tokenize), self.process_sample(text, tokenizer, tokenize)
        return [prompt], [text]


class TestDataset(PromptReader):
    PATH = '/root/data/test.json'
    assert_str = "make sure to set PATH for wikipedia data_utils/corpora.py"

    def process_line(self, data, tokenizer, tokenize):
        prompt, text = data['prompt'], data['text']
        prompt, text = self.process_sample(prompt, tokenizer, tokenize), self.process_sample(text, tokenizer, tokenize)
        return [prompt], [text]


class OpenWebText(PromptReader):
    PATH = '/dataset/fd5061f6/english_data/openwebtext2'
    assert_str = "make sure to set PATH for openwebtext data_utils/corpora.py"

    def __init__(self, *args, **kwargs):
        import fasttext
        super().__init__(*args, **kwargs)
        self.model = fasttext.load_model('/dataset/fd5061f6/english_data/lid.176.bin')
        print_rank_0("Load language detection model")

    def process_line(self, data, tokenizer, tokenize):
        text = data['text']
        if len(text) > 100:
            lang = self.model.predict(text.replace('\n', ''))[0][0]
            if lang == '__label__en':
                prompt, text = self.process_sample("", tokenizer, tokenize), self.process_sample(text, tokenizer,
                                                                                                 tokenize)
                return [prompt], [text]
        return [], []


class CCNews(PromptReader):
    PATH = "/mnt/cc_news.json"
    assert_str = "make sure to set PATH for cc-news data_utils/corpora.py"

    def process_line(self, data, tokenizer, tokenize):
        text = ""
        title = data.get("title", None)
        description = data.get("description", None)
        maintext = data.get("maintext", None)
        if title:
            text += title.strip() + " "
        if description and (not maintext or not maintext.startswith(description)):
            text += description.strip() + " "
        if maintext:
            text += maintext
        if len(text) > 100:
            prompt, text = self.process_sample("", tokenizer, tokenize), self.process_sample(text, tokenizer, tokenize)
            return [prompt], [text]
        else:
            return [], []


class BertData(PromptReader):
    is_json = False
    PATH = '/dataset/fd5061f6/english_data/wikibook'

    def process_line(self, data, tokenizer, tokenize):
        if data:
            prompt, text = "", data
            prompt, text = self.process_sample(prompt, tokenizer, tokenize), self.process_sample(text, tokenizer,
                                                                                                 tokenize)
            return [prompt], [text]
        else:
            return [], []


class Pile(PromptReader):
    is_json = True
    PATH = "/mnt/train"
    filtered_sources = ["Github", "StackExchange", "DM Mathematics", "Ubuntu IRC", "EuroParl", "YoutubeSubtitles",
                        "Enron Emails"]
    downsample_sources = {"PubMed Central": 0.3, "ArXiv": 0.3, "FreeLaw": 0.3}

    def print_info(self, info):
        total_dict = defaultdict(int)
        while True:
            try:
                source_dict = info.get(block=False)
                for source, length in source_dict.items():
                    total_dict[source] += length
            except Empty:
                break
        print_rank_0(total_dict)

    def tokenize_worker(self, input, output, info, tokenizer, tokenize):
        source_dict = defaultdict(int)
        for row in iter(input.get, 'STOP'):
            row = row.rstrip()
            if row:
                if self.is_json:
                    row = json.loads(row)
                prompts, texts, source = self.process_line(row, tokenizer, tokenize)
                length = 0
                for prompt, text in zip(prompts, texts):
                    length += len(text)
                    output.put((prompt, text))
                if source:
                    source_dict[source] += length
        output.put("COMPLETE")
        info.put(source_dict)

    def process_line(self, data, tokenizer, tokenize):
        source = data["meta"].get("pile_set_name", None)
        text = data.get("text", None)
        if source and text:
            if source in self.filtered_sources:
                return [], [], None
            elif source in self.downsample_sources and random.random() > self.downsample_sources[source]:
                return [], [], None
            else:
                prompt, text = self.process_sample("", tokenizer, tokenize), self.process_sample(text, tokenizer,
                                                                                                 tokenize)
                return [prompt], [text], source
        else:
            return [], [], None


class Stories(PromptReader):
    is_json = True
    PATH = "/dataset/fd5061f6/english_data/stories_31G.jsonl"

    def process_line(self, data, tokenizer, tokenize):
        text = data.get("text", None)
        if text:
            prompt, text = self.process_sample("", tokenizer, tokenize), self.process_sample(text, tokenizer,
                                                                                             tokenize)
            return [prompt], [text]
        else:
            return [], []


class BertBaseData(BertData):
    PATH = '/root/data/formatted_one_article_per_line'


class BertLargeData(BertData):
    PATH = '/dataset/c07bd62b/cognitive/zhengxiao/formatted_one_article_per_line_large'


class WuDaoCorpus(PromptReader):
    PATH = "/dataset/fd5061f6/chinese_data/WuDao"
    is_json = False
    reserve_punct = True
    split_row = False

    def process_line(self, item, tokenizer, tokenize):
        prompts, texts = [], []
        text = ""
        title = item.get("title", None)
        content = item.get("content", None)
        if title:
            text += title.strip() + " "
        if content:
            text += content
        if len(text) > 100:
            prompt, text = self.process_sample("", tokenizer, tokenize), self.process_sample(text, tokenizer,
                                                                                             tokenize)
            prompts.append(prompt)
            texts.append(text)
        return prompts, texts


class KorDataset_json(PromptReader):
    is_json = True
    reserve_punct = True
    split_row = False
    category = None

    def tokenize_worker(self, input, output, info, tokenizer, tokenize):
        # iterate through json['document']
        for row in iter(input.get, 'STOP'):
            # read json
            for paragraph in row[self.category]:
                prompts, texts = self.process_line(paragraph['form'], tokenizer, tokenize)
                    # print(f'texts - {texts}')
                for prompt, text in zip(prompts, texts):
                    output.put((prompts, text))
        output.put("COMPLETE")
    
    # handling one line not document??
    def process_line(self, data, tokenizer, tokenize):
        # prompt is not implemented
        prompts, texts = [], []
        
        # fix bugs
        # temp -> prompt: "" => data (because of empty file error)
        prompt, text = [], self.process_sample(data, tokenizer, tokenize)
        prompts.append(prompt)
        texts.append(text)
        # return [], [text]
        return [prompt], texts


class KorDataset(PromptReader):
    is_json=False
    reserve_punct = True  # whether punctuation is in the vocab
    split_row = True
    def process_line(self, data, tokenizer, tokenize):
        text = data.strip('\n').strip()
        if text:
            prompt, text = self.process_sample("", tokenizer, tokenize), self.process_sample(text, tokenizer, tokenize)
            return [prompt], [text]
        else:
            return [], [] 


class Online_Review(KorDataset):
    PATH = "/data/sgahn/online_review"
    

class NIKL_Daily_conversation(KorDataset):
    PATH = "/data/sgahn/NIKL_DAILY_CONVERSATION_2020_v1.2_text"
    # category = 'utterance'
    
    # based on other classes,,
    # prompt -> return [[]]
    # text -> clean the text

class NIKL_KParlty(KorDataset):
    PATH ="/data/sgahn/NIKL_KParlty_2021_v1.0_text"
    # category = 'utterance'


class NIKL_Messenger(KorDataset):
    PATH = "/data/sgahn/NIKL_MESSENGER_v2.0_text"


class NIKL_Newspaper(KorDataset):
    PATH = "/data/sgahn/NIKL_NEWSPAPER_text"
    # category = 'paragraph'


class NIKL_Newspaper_2020(KorDataset):
    PATH = "/data/sgahn/NIKL_NEWSPAPER_2020_text"


class NIKL_Newspaper_2021(KorDataset):
    PATH = "/data/sgahn/NIKL_NEWSPAPER_2021_v1.0_text"


class NIKL_Om(KorDataset):
    PATH = "/data/sgahn/NIKL_OM_2021_v1.0_text"


class NIKL_Spoken(KorDataset):
    PATH = "/data/sgahn/NIKL_SPOKEN_text"


class NIKL_Written(KorDataset):
    PATH = "/data/sgahn/NIKL_WRITTEN_text"

   

NAMED_CORPORA = {
    'wikipedia': wikipedia,
    'wikipedia-key': KeyReader,
    'openwebtext': OpenWebText,
    "zhihu": zhihu,
    "zhidao": zhidao,
    "baike": baike,
    "test": TestDataset,
    'wikibook': BertData,
    "bert-base": BertBaseData,
    "bert-large": BertLargeData,
    'cc-news': CCNews,
    'pile': Pile,
    'stories': Stories,
    'wudao': WuDaoCorpus,
    'nikl_daily': NIKL_Daily_conversation,
    'nikl_kparlty': NIKL_KParlty,
    'nikl_messenger': NIKL_Messenger,
    'nikl_newspaper': NIKL_Newspaper,
    'nikl_newspaper_2020': NIKL_Newspaper_2020,
    'nikl_newspaper_2021': NIKL_Newspaper_2021,
    'nikl_om' : NIKL_Om,
    'nikl_spoken': NIKL_Spoken,
    'nikl_written': NIKL_Written,
    'online_review': Online_Review,
}
