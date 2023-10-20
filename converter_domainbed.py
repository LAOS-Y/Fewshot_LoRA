import numpy as np
import os.path as osp
import json
import DomainBed.domainbed.datasets as dbdatasets

import DomainBed.domainbed.lib.misc as misc
from DomainBed.domainbed.lib.fast_data_loader import InfiniteDataLoader

def get_subdatasets(dataset, class_keys):
    base_keys, open_keys = [], []
    for i, (_, label) in enumerate(dataset.samples):
        if label in class_keys:
            base_keys.append(i)
        else:
            open_keys.append(i)
    base_dataset = misc._SplitDataset(dataset, base_keys)
    open_dataset = misc._SplitDataset(dataset, open_keys)
    return base_dataset, open_dataset


def get_domainbed_fewshot_datasets(dataset_name, root, targets, shot_num, split=0):
    def is_valid_file_train(img_path):
        dir_path, img_filename = osp.split(img_path)
        filenames = json.load(open(osp.join(dir_path, "shuffled_filenames.json")))
        assert (split + 1) * shot_num < len(filenames), (dir_path, len(filenames))
        filenames = filenames[split * shot_num: (split + 1) * shot_num]
        return img_filename in filenames

    # TODO: should we have the val set when few-shot
    def is_valid_file_val(img_path):
        dir_path, img_filename = osp.split(img_path)
        filenames = json.load(open(osp.join(dir_path, "shuffled_filenames.json")))
        assert (split + 2) * shot_num < len(filenames), (dir_path, len(filenames))
        filenames = filenames[(split + 1) * shot_num: (split + 2) * shot_num]
        return img_filename in filenames

    train_datasets = vars(dbdatasets)[dataset_name](
        root, targets, hparams={
            "data_augmentation": True,
            "is_valid_file": is_valid_file_train
        }
    )

    # set target = list(range(len(train_datasets))) so that all domains
    # use test-time aug as val sets
    val_datasets = vars(dbdatasets)[dataset_name](
        root, list(range(len(train_datasets))), hparams={
            "data_augmentation": True,
            "is_valid_file": is_valid_file_val
        }
    )

    # set target = list(range(len(train_datasets))) so that all domains
    # use test-time aug as val sets
    test_datasets = vars(dbdatasets)[dataset_name](
        root, targets, hparams={
            "data_augmentation": True,
            "is_valid_file": None
        }
    )

    # Remove wrong domains
    train_datasets = [d for (i, d) in enumerate(train_datasets) if i not in targets]
    val_datasets = [d for (i, d) in enumerate(val_datasets) if i not in targets]
    test_datasets = [d for (i, d) in enumerate(test_datasets) if i in targets]

    return train_datasets, val_datasets, test_datasets, train_datasets[0].classes


def get_domainbed_datasets(dataset_name, root, targets, holdout=0.2, open_ratio=0):
    assert dataset_name in vars(dbdatasets)
    hparams = {"data_augmentation": True}
    datasets = vars(dbdatasets)[dataset_name](root, targets, hparams)
    class_names = datasets[0].classes
    if open_ratio > 0:
        # Sample subclasses
        keys = list(range(len(class_names)))
        base_class_keys = keys[:int((1 - open_ratio) * len(keys))]
        base_class_names = [class_name for i, class_name in enumerate(class_names) if i in base_class_keys]
        open_class_names = [class_name for class_name in class_names if class_name not in base_class_names]
        in_bases, in_opens, out_bases, out_opens = [], [], [], []
        for env_i, env in enumerate(datasets):
            base_env, open_env = get_subdatasets(env, base_class_keys)
            out_base, in_base = misc.split_dataset(base_env, int(len(base_env) * holdout), misc.seed_hash(0, env_i, "base"))
            out_open, in_open = misc.split_dataset(open_env, int(len(open_env) * holdout), misc.seed_hash(0, env_i, "open"))
            in_bases.append(in_base)
            in_opens.append(in_open)
            out_bases.append(out_base)
            out_opens.append(out_open)
        train_datasets = [d for (i, d) in enumerate(in_bases) if i not in targets]
        val_datasets = [d for (i, d) in enumerate(out_bases) if i not in targets]
        test_datasets = [d for (i, d) in enumerate(in_bases) if i in targets] + [d for (i, d) in enumerate(out_bases) if i in targets]
        open_datasets = [d for (i, d) in enumerate(in_opens) if i in targets] + [d for (i, d) in enumerate(out_opens) if i in targets]
        return train_datasets, val_datasets, test_datasets, open_datasets, base_class_names, open_class_names
    else:
        in_splits, out_splits = [], []
        for env_i, env in enumerate(datasets):
            out, in_ = misc.split_dataset(env,
                int(len(env) * holdout),
                misc.seed_hash(0, env_i))
            in_splits.append(in_)
            out_splits.append(out)
        train_datasets = [d for (i, d) in enumerate(in_splits) if i not in targets]
        # Note by Siwei: the val set still uses the same augmentation on the train
        # set while it should align with the test set
        val_datasets = [d for (i, d) in enumerate(out_splits) if i not in targets]
        # Note by Siwei: Not all the data from the target domain is used as the test set
        test_datasets = [d for (i, d) in enumerate(out_splits) if i in targets]
        return train_datasets, val_datasets, test_datasets, class_names

def get_forever_iter(datasets, batch_size, num_workers):
    iters = [InfiniteDataLoader(dataset, None, batch_size, num_workers) for dataset in datasets]
    return zip(*iters)
