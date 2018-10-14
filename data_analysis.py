from collections import defaultdict

import matplotlib.pyplot as plt
import cv2
from data import Data_handler
import numpy as np

class Data_analysis():
    @staticmethod
    def create_templates(train_split):
        """
        Creates the templates.
        TODO: Write more info here later
        :param train_split: the train split
        :return:
        """
        signs_by_type = defaultdict(list)
        count = 0
        for image in train_split:
            im = cv2.imread(image.img)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            # mask = cv2.imread(image.msk)
            # im = cv2.bitwise_and(im, im, mask=mask[:,:,0].astype(dtype=np.uint8)) # should we remove the background?

            for ann in image.annotations:
                count += 1
                bbox = ann[0]

                tall = False
                height = int(bbox[2]) - int(bbox[0])
                width = int(bbox[3]) - int(bbox[1])

                if height > width:
                    diff = height - width
                    bbox[3] = int(bbox[3]) + diff
                    bbox[1] = int(bbox[1]) - diff
                    if (height / width) > 1.1:
                        tall = True
                else:
                    diff = width - height
                    bbox[2] = int(bbox[2]) + diff
                    bbox[0] = int(bbox[0]) - diff

                sign = im[int(bbox[0]): int(bbox[2]), int(bbox[1]): int(bbox[3]), :]
                sign = cv2.resize(sign, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
                if ann[1] == 'F':
                    key_extension = '_1' if tall else '_2'
                    signs_by_type[ann[1]+key_extension].append(cv2.cvtColor(sign, cv2.COLOR_BGR2GRAY))
                else:
                    signs_by_type[ann[1]].append(cv2.cvtColor(sign, cv2.COLOR_BGR2GRAY))
                plt.imsave("./data/" + ann[1] + "/" + str(count) + ".png", sign)

        maxcomp = 3
        for key in signs_by_type:
            mean = np.asarray(signs_by_type[key]).mean(axis=0)
            plt.imsave("./data/templates/mean_" + key + ".png", mean.astype(dtype=int), cmap='gray')

            mean, eigenVectors = cv2.PCACompute(np.asarray(signs_by_type[key]).reshape(len(signs_by_type[key]), 10000),
                                                mean=None, maxComponents=maxcomp)
            for i in range(maxcomp):
                plt.imsave("./data/templates/" + "eVector_" + key + "_" + str(i) + ".png",
                           eigenVectors[i].reshape(100, 100), cmap='gray')

        return

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
                    image   = cv2.imread(image_instance.img)
                    #cv2.imshow('image', image)
                    #cv2.waitKey(0)
                    mask_im = cv2.imread(image_instance.msk, 0)

                    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                    histH = cv2.calcHist([hsv_image], [0], mask_im, [256], [0, 256])
                    histS = cv2.calcHist([hsv_image], [1], mask_im, [256], [0, 256])
                    histV = cv2.calcHist([hsv_image], [2], mask_im, [256], [0, 256])
                    sign_type = ann[1]
                    if sign_type in histograms_by_signal:
                        histograms_by_signal[sign_type].append((histH, histS, histV))
                    else:
                        histograms_by_signal[sign_type] = [(histH, histS, histV)]
                    #print(sign_type)
                    #plt.plot(histH)
                    #plt.show()

            colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
            for sign_type in histograms_by_signal:
                count = 0
                for i, hist in enumerate(histograms_by_signal[sign_type]):
                    sum_hist = hist[0].sum()
                    plt.subplot(3, 2, 1)
                    plt.plot(hist[0] / sum_hist, color=colors[i % len(colors)])
                    plt.title("Hue  " + sign_type + ": " + Data_handler.parse_sign_type(sign_type))
                    plt.subplot(3, 2, 3)
                    plt.plot(hist[1] / sum_hist, color=colors[i % len(colors)])
                    plt.title("Saturation  " + sign_type + ": " + Data_handler.parse_sign_type(sign_type))
                    plt.subplot(3, 2, 5)
                    plt.plot(hist[2] / sum_hist, color=colors[i % len(colors)])
                    plt.title("Value  " + sign_type + ": " + Data_handler.parse_sign_type(sign_type))
                    count += 1
                    if (i == 0):
                        sumHist = [hist[0] / sum_hist, hist[1] / sum_hist, hist[2] / sum_hist]
                    else:
                        sumHist[0] += hist[0] / sum_hist
                        sumHist[1] += hist[1] / sum_hist
                        sumHist[2] += hist[2] / sum_hist

                plt.subplot(3, 2, 2)
                plt.plot(sumHist[0] / count)
                plt.title("Hue Sum " + sign_type + ": " + Data_handler.parse_sign_type(sign_type))
                plt.subplot(3, 2, 4)
                plt.plot(sumHist[1] / count)
                plt.title("Saturation  " + sign_type + ": " + Data_handler.parse_sign_type(sign_type))
                plt.subplot(3, 2, 6)
                plt.plot(sumHist[2] / count)
                plt.title("Value  " + sign_type + ": " + Data_handler.parse_sign_type(sign_type))

                plt.show()
            return histograms_by_signal


