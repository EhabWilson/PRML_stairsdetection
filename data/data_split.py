import json
import random
from pathlib import Path

DATA_DIR = '../dataset/public'
SEED = 123
FILE_PREFIX = 'public'
SPLIT_RATIO = (7, 1, 2)

random.seed(SEED)
sum_ratio = sum(SPLIT_RATIO)
normed_ratio = [ratio / sum_ratio for ratio in SPLIT_RATIO]
cusum = 0.
cdf = [0] * len(normed_ratio)
for i, ratio in enumerate(normed_ratio):
    cusum += ratio
    cdf[i] = cusum

train_set, valid_set, test_set = [], [], []
for img_dir in Path(DATA_DIR).rglob('*.jpg'):
    p = random.random()
    img_dir = str(img_dir)
    label = 0 if 'no_stairs' in img_dir else 1
    if p < cdf[0]:
        train_set.append({'dir': img_dir, 'label':label})
    elif p < cdf[1]:
        valid_set.append({'dir': img_dir, 'label':label})
    else:
        test_set.append({'dir': img_dir, 'label':label})

with open(f'data/{FILE_PREFIX}_train.json', 'wt') as f:
    json.dump(train_set, f, indent=4)

with open(f'data/{FILE_PREFIX}_valid.json', 'wt') as f:
    json.dump(valid_set, f, indent=4)

with open(f'data/{FILE_PREFIX}_test.json', 'wt') as f:
    json.dump(test_set, f, indent=4)