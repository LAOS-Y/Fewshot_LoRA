import os
import os.path as osp
import json

import argparse
import random
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Generate shuffled_filename.json '
                                             'for few-shot training stability')
# dataset parameters
parser.add_argument('root', metavar='DIR',
                    help='root path of dataset')
args = parser.parse_args()

root = args.root

for domain in tqdm(os.listdir(root)):
    if not osp.isdir(osp.join(root, domain)):
        continue
    for category in tqdm(os.listdir(osp.join(root, domain))):
        path = osp.join(root, domain, category)
        filenames = sorted(os.listdir(path))
        random.shuffle(filenames)

        output_path = osp.join(root, domain, category, "shuffled_filenames.json")
        print(output_path, len(filenames))
        json.dump(filenames, open(output_path, mode="w"))
