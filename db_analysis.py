import os
#  import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from collections import defaultdict
#  import sys


def read_gt(gt_file_path):
    """
    This function reads every bounding box of the ground truth and stores the data in a list. Each element of the list
    is an annotation of the ground truth, and has the following structure:
    gt: [image id, [tly, tlx, bry, brx], sign type, aspect ratio]
    :param gt_file_path: path to the gt
    :return: the list with the annotations
    """

    files_gt = os.listdir(gt_file_path)
    gt = []

    for filename in files_gt:
        id_ = filename.replace("gt.", "").replace(".txt", "")
        with open(gt_file_path + filename, "r") as file:
            for line in file.readlines():
                content = line.rstrip("\n").split(" ")
                bbox = list(map(float, content[:4]))
                aspect_ratio = (bbox[3] - bbox[1]) / (bbox[2] - bbox[0])  # aspect ratio of the bbox
                gt.append([id_, bbox, content[4], aspect_ratio])

    return gt


def visualize(gt, image_file_path):
    """
    This is just a function I used to make sure I got the parameters of the bounding boxes right. It plots an image
    with a bounding box where the signs are.
    :param gt:
    :param image_file_path:
    :return:
    """
    for ann in gt:
        #  image = cv2.imread(gt_file_path + ann[0] + ".jpg")
        print(image_file_path + ann[0] + ".jpg")
        image = plt.imread(image_file_path + ann[0] + ".jpg")
        _, ax = plt.subplots(1)
        ax.imshow(image)
        rect = pat.Rectangle([int(ann[1][1]), int(ann[1][0])], int(ann[1][3] - ann[1][1]),
                             int(ann[1][2] - ann[1][0]), linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        plt.show()


def stats(gt, mask_file_path):
    """
    This function extracts stats about the annotations.
    :param gt: gt annotations
    :param mask_file_path: path to the masks' folder
    :return:
    """

    signs = []
    max_area = 0
    min_area = float('inf')
    filling_ratios = defaultdict(list)  # the key is the sign type and the value is a list with the filling ratios

    for ann in gt:
        mask = plt.imread(mask_file_path + "mask." + ann[0] + ".png")
        bbox = ann[1]

        # filling ratio
        pixel_count = 0
        for x in range(int(bbox[0]), int(bbox[2])):
            for y in range(int(bbox[1]), int(bbox[3])):
                if mask[x, y]:
                    pixel_count += 1
        filling_ratio = pixel_count / ((int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1])))
        filling_ratios[ann[2]].append(filling_ratio)

        # max/min area
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        if area > max_area:
            max_area = area
        if area < min_area:
            min_area = area

        # sign type count
        signs.append(ann[2])

    sign_types = set(signs)
    sign_count = []
    for sign_type in sign_types:
        sign_count.append([sign_type, signs.count(sign_type)])

    return sign_count, max_area, min_area, filling_ratios


"""
        for x in range(int(bbox[0]), int(bbox[2])):
            for y in range(int(bbox[1]), int(bbox[3])):
                if mask[x, y]:
                    sys.stdout.write("1")
                else:
                    sys.stdout.write("0")
            sys.stdout.write("\n")
"""

def sign_counter(gt):
    """
    Just a small function that counts the signs
    :param gt: gt annotations
    :return: signs count as a dict
    """
    signs = []

    for ann in gt:
        signs.append(ann[2])

    sign_types = set(signs)
    sign_count = []
    for sign_type in sign_types:
        sign_count.append([sign_type, signs.count(sign_type)])

    return sign_count


def main():
    gt_file_path = "./train/gt/"
    image_file_path = "./train/"
    mask_file_path = "./train/mask/"

    gt = read_gt(gt_file_path)
    sign_count, max_area, min_area, filling_ratios = stats(gt, mask_file_path)

    print("Sign type count:")
    for sign in sign_count:
        print(sign[0], ": ", sign[1])
    print("max area: ", max_area)
    print("min area: ", min_area)

    print("filling ratios:")
    for key in filling_ratios.keys():
        print(key + ": " + str(sum(filling_ratios[key])/len(filling_ratios[key])))

    #  exit(0)


if __name__ == "__main__":
    main()
