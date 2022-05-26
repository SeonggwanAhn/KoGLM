import sys
import os
import json
"""
argument 1: directory of runs
argument 2: path of ratings_test.txt
"""

result_path = os.path.join(sys.argv[1], "nsmc-test.jsonl")
try:
    test_path = sys.argv[2] 
except IndexError:
    test_path = "/data/sgahn/superglue/nsmc/ratings_test.txt"

with open(result_path, 'r') as f:
    result_lines = f.readlines()

with open(test_path, 'r') as f:
    test_lines = f.readlines()

cnt = 0
for res, te in zip(result_lines, test_lines[1:]):
    data = json.loads(res)
    _, doc, label = te.strip().split('\t')
    if str(data['label']) != str(label):
        print(doc)
        cnt += 1
print(cnt)
