import pickle
import os
import matplotlib.pyplot as plt


# note that now the bbox and the sign type are stored as a tuple in a list called annotations. This makes iterating
# over the annotations easier
# TODO: self.types should be initialized using some function in data_analysis

class Instance():
    def __init__(self, img_, msk_, img_id_):
        self.img = img_            # numpy array  ==> rgb
        self.msk = msk_            # numpy array  ==> binary image 0 or 1
        self.annotations = []      # [ ([tly, tlx, bry, brx], sign_type), ... ]
        self.img_id = img_id_      # srting


class Data_handler():
    """
    Handles all the data related to the images and its annotations
    self.train_set contains a list with all the data of the images (image+mask+(bounding box, sign type)+image id)
    self.valid_set and self.test_set only are lists with the ids of the images in these splits as we don't need
    the annotations.
    """
    def __init__(self, train_dir='./train/', test_dir = './test/'):
        self.train_set = []      # [Instance(), ...]
        self.valid_set = []      # [id, ...]
        self.test_set  = []      # [id, ...]
        self.test_set_ids = []
        self.valid_set_ids = []
        self.train_set_ids = []
        self.types     = ['A','B','C','D','E','F'] 
        self.train_dir = train_dir
        self.test_dir  = test_dir

    def read_all(self):
        with open("./data/val_split.pkl", "rb") as f:
            valid_ids = pickle.load(f)
            self.valid_set_ids = valid_ids

        with open("./data/train_split.pkl", "rb") as f:
            train_ids = pickle.load(f)
            self.train_set_ids = train_ids

        with open("./data/test_split.pkl", "rb") as f:
            test_ids  = pickle.load(f)
            self.test_set_ids = test_ids

        gt_file_path = self.train_dir + "gt/"

        for id_ in train_ids:
            filename = id_ + ".txt"
            image = (self.train_dir + id_ + ".jpg")
            mask  = (self.train_dir + "mask/" + "mask." + id_ + ".png")
            ann   = Instance(image, mask, id_)
            with open(gt_file_path + "gt." + filename, "r") as file:
                for line in file.readlines():
                    content = line.rstrip("\n").split(" ")
                    bbox = list(map(float, content[:4]))
                    ann.annotations.append((bbox, content[4]))
            self.train_set.append(ann)

        for id_ in valid_ids:
            filename = id_ + ".txt"
            image = (self.train_dir + id_ + ".jpg")
            mask  = (self.train_dir + "mask/" + "mask." + id_ + ".png")
            ann   = Instance(image, mask, id_)

            with open(gt_file_path + "gt." + filename, "r") as file:
                for line in file.readlines():
                    content = line.rstrip("\n").split(" ")
                    bbox = list(map(float, content[:4]))
                    ann.annotations.append((bbox, content[4]))
            self.valid_set.append(ann)

        for id_ in test_ids:
            filename = id_ + ".txt"
            image = (self.test_dir + id_ + ".jpg")
            mask  = None
            ann   = Instance(image, mask, id_)

            self.test_set.append(ann)

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
