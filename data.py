import pickle
import os
import matplotlib.pyplot as plt


# note that now the bbox and the sign type are stored as a tuple in a list called annotations. This makes iterating
# over the annotations easier
# TODO: self.types should be initialized using some function in data_analysis

class instance():
    img         = None       # numpy array  ==> rgb
    msk         = None       # numpy array  ==> binary image 0 or 1
    annotations = None       # [ ([tly, tlx, bry, brx], sign_type), ... ]
    img_id      = None       # srting

    def __init__(self, img_, msk_, img_id_):
        self.img = img_
        self.msk = msk_
        self.annotations = []
        self.img_id = img_id_


class data_handler():
    """
    Handles all the data related to the images and its annotations
    self.train_set contains a list with all the data of the images (image+mask+(bounding box, sign type)+image id)
    self.valid_set and self.test_set only are lists with the ids of the images in these splits as we don't need
    the annotations.
    """
    def __init__(self, train_dir='./train/'):
        self.train_set = []  # [instance(), ...]
        self.valid_set = []  # [id, ...]
        self.test_set = []  # [id, ...]
        self.types = []  # ['A','B','C','D','E','F']

        with open("./data/val_split.pkl", "rb") as f:
            self.valid_set = pickle.load(f)

        with open("./data/train_split.pkl", "rb") as f:
            train_ids = pickle.load(f)

        with open("./data/test_split.pkl", "rb") as f:
            self.test_set = pickle.load(f)

        gt_file_path = train_dir + "gt/"

        for id_ in train_ids:
            filename = id_ + ".txt"

            image = (train_dir + id_ + ".jpg")
            mask = (train_dir + "mask/" + "mask." + id_ + ".png")
            ann = instance(image, mask, id_)
            with open(gt_file_path + "gt." + filename, "r") as file:
                for line in file.readlines():
                    content = line.rstrip("\n").split(" ")
                    bbox = list(map(float, content[:4]))
                    ann.annotations.append((bbox, content[4]))
            self.train_set.append(ann)

    def read_all(self):
        return self.train_set, self.valid_set, self.test_set

    @staticmethod
    def parse_sign_type(sign_type):
        if sign_type == 'A':
            return "Red Triangle"
        elif sign_type == 'B':
            return "Yield"
        elif sign_type == 'C':
            return "Red Circle"
        elif sign_type == 'D':
            return "Blue Circle"
        elif sign_type == 'E':
            return "Red and Blue Circle"
        elif sign_type == 'F':
            return "Blue Square"
        return "Unknown sign type"
