import argparse
from tokenizers import BertWordPieceTokenizer

parser = argparse.ArgumentParaser()

parser.add_argument("--corpus_file", type=str)
parser.add_argument("--vocab_sizes", type=int, default=32000)
parser.add_argument("--limit_alphabet", type=int, default=6000)

args = parser.parse_args()


tokenizer = BertWordPieceTokenizer(
		vocab_file=None,
		clean_text=True,
		handle_chinese_chars=True,
		strip_accents=False,
		lowercase=False,
		wordpieces_prefix="##"
)

toke
