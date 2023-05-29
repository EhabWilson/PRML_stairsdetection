import json
import random
import shutil
import os
from pathlib import Path

def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
DATA_DIR = "."
# SPLIT_DIR = "../dataset/public_split"   # dir to store splited data
# TRAIN_DIR = os.path.join(SPLIT_DIR, "train")
# VALID_DIR = os.path.join(SPLIT_DIR, "valid")
# TEST_DIR = os.path.join(SPLIT_DIR, "test")
# makedir(SPLIT_DIR)
# makedir(TRAIN_DIR)
# makedir(VALID_DIR)
# makedir(TEST_DIR)

SEED = 123
FILE_PREFIX = 'public'
JSON_DIR = "../../codes/data"
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

with open(f'{JSON_DIR}/{FILE_PREFIX}_train.json', 'wt') as f:
    json.dump(train_set, f, indent=4)

with open(f'{JSON_DIR}/{FILE_PREFIX}_valid.json', 'wt') as f:
    json.dump(valid_set, f, indent=4)

with open(f'{JSON_DIR}/{FILE_PREFIX}_test.json', 'wt') as f:
    json.dump(test_set, f, indent=4)
