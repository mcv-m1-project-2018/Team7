from collections import defaultdict

import matplotlib.pyplot as plt
import cv2


class Data_analysis():
    @staticmethod
    def shape_analysis(train_split):
        """
        This function extracts stats about shape of the annotations.
        :param train_split: train_split annotations
        :return: sign_count, max_area, min_area, filling_ratios, max_aspect_ratio, min_aspect_ratio
        """

        signs = []
        max_area = 0
        min_area = float('inf')
        max_aspect_ratio = 0
        min_aspect_ratio = float('inf')
        filling_ratios = defaultdict(list)  # the key is the sign type and the value is a list with the filling ratios

        for image in train_split:
            mask = plt.imread(image.msk)
            for ann in image.annotations:
                bbox = ann[0]

                # filling ratio
                pixel_count = 0
                for x in range(int(bbox[0]), int(bbox[2])):
                    for y in range(int(bbox[1]), int(bbox[3])):
                        if mask[x, y]:
                            pixel_count += 1
                filling_ratio = pixel_count / ((int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1])))
                filling_ratios[ann[1]].append(filling_ratio)

                # max/min area
                area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
                if area > max_area:
                    max_area = area
                if area < min_area:
                    min_area = area

                # sign type count
                signs.append(ann[1])

                # aspect ratio
                aspect_ratio = (bbox[3] - bbox[1]) / (bbox[2] - bbox[0])
                if aspect_ratio > max_aspect_ratio:
                    max_aspect_ratio = aspect_ratio
                if aspect_ratio < min_aspect_ratio:
                    min_aspect_ratio = aspect_ratio

        sign_types = set(signs)
        sign_count = {}
        for sign_type in sign_types:
            sign_count[sign_type] = signs.count(sign_type)

        for key in filling_ratios.keys():
            filling_ratios[key] = [sum(filling_ratios[key]) / len(filling_ratios[key])]

        return sign_count, max_area, min_area, filling_ratios, max_aspect_ratio, min_aspect_ratio

    @staticmethod
    def color_analysis(train_split):
            """
            This function extracts information about the color.
            :param train_split: list of instances
            :return:
            """
            histograms_by_signal = {}
            for image_instance in train_split:
                for ann in image_instance.annotations:
                    image   = plt.imread(image_instance.img)
                    mask_im = plt.imread(image_instance.msk)

                    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                    histH = cv2.calcHist([hsv_image], [0], mask_im, [256], [0, 256])
                    histS = cv2.calcHist([hsv_image], [1], mask_im, [256], [0, 256])
                    histV = cv2.calcHist([hsv_image], [2], mask_im, [256], [0, 256])
                    sign_type = ann[1]
                    if sign_type in histograms_by_signal:
                        histograms_by_signal[sign_type].append((histH, histS, histV))
                    else:
                        histograms_by_signal[sign_type] = [(histH, histS, histV)]

