import os
import random
from db_analysis import read_gt, stats
from collections import defaultdict


def db_split(train_prop, db_file_path):
    """
    Splits the database according to the train proportion. I've tried to assign the same proportion of signs to each
    one of the splits. This has been a bit complicated because some images have more than one sign (so when it added
    one of the images to the train split it added 1 or 2 more besides, and this spoiled a bit the proportions).
    To assign the image to the training split it checks the type of signals it contains, from the more rare types to
    less rare (line 45). This way rare signs have priority over more common signs, which helps to split the data. The
    actual proportions of each type of sign changes with execution because the sorting (I guess?), but here's an
    example with a 70-30 split:
    ['E', 26]   0.68
    ['F', 87]   0.72
    ['B', 9]   0.64
    ['C', 29]   0.61
    ['A', 75]   0.72
    ['D', 49]   0.69
    This function if a bit convoluted but it works.
    :param train_prop: proportion of the training split
    :param db_file_path: path to the training data
    :return: train: a list with the ids of the images in the training set
    :return: val: a list with the ids if the images in the validation set
    """
    gt = read_gt(db_file_path + "gt/")
    sign_count, _, _, _ = stats(gt, db_file_path + "mask/")
    sign_count.sort(key=lambda ls: ls[1])

    gt_files = os.listdir(db_file_path + "gt/")
    # random.shuffle(gt_files)

    train_split = defaultdict(list)

    ids = []
    # this is necessary to obtain the validation set at the end
    for filename in gt_files:
        id_ = filename.replace("gt.", "").replace(".txt", "")
        ids.append(id_)

    count = 0
    for filename in gt_files:
        id_ = filename.replace("gt.", "").replace(".txt", "")
        signs_current = []
        with open(db_file_path + "gt/" + filename) as f:
            for line in f.readlines():
                line = line.split()
                signs_current.append(line[4])

        for sign in sign_count:
            if len(train_split[sign[0]]) < int(sign[1] * train_prop) and sign[0] in signs_current:
                count += 1
                train_split[sign[0]].append(id_)
                break

        if count >= len(gt_files)*train_prop:
            break

    train = []
    for element in train_split:
        train.extend(train_split[element])

    val = list(set(ids).difference(train))

    return train, val


if __name__ == "__main__":
    db_split(0.70, "./train/")

