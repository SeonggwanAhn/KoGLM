from collections import Counter
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "NanumGothic"
tkn = AutoTokenizer.from_pretrained("/home/sgahn/koelectra-base-v3-discriminator")

corpus = "/data/sgahn/superglue/nsmc/ratings.txt"
with open(corpus, 'r') as f:
    lines = f.readlines()


ignore = ['.',  '!', ',', '##도', '##다', '?', '##과', '##하다', '다', '##듯', '##리', '##하게', \
        '##이', '~', '##가', '##는', '##만', '##지', '##의', '##한', '##야', '##에서', '##할', '##점이'\
        '##고', '##은', '##에', '##게', '##나', '##음', '이', '##네', '##네요', '##을', '##하고', ';',\
        '아', '##서', '##거', '^', '##라', '##들', '##아', '##기', '그', '##는데', '##를', '##어',\
        '##화', '"']

for i in range(1000):
    ignore.append(str(i))
    ignore.append('##'+str(i))


tokens = list()
for line in lines[1:]:
    token = tkn.tokenize(line)
    tokens.extend([t for t in token if t not in ignore and not t.startswith('##')])


tkn_cnter = Counter(tokens)
# print(tkn_cnt.most_common())

k = 10
top_k = tkn_cnter.most_common(k)
word_freq = pd.DataFrame(top_k, columns=["토큰", "빈도"])

fig, ax = plt.subplots(figsize=(8, 5))

word_freq.sort_values(by='빈도').plot.barh(x="토큰", ax=ax, legend=False, fontsize=15)
ax.set_ylabel('토큰', fontsize=15)
ax.set_xlabel('빈도 수', fontsize=15)
plt.show()
