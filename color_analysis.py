from db_analysis import read_gt
import cv2
import numpy as np
import matplotlib.pyplot as plt


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

def red_test(gt, mask_file_path, image_file_path):
    for ann in gt:
        image = cv2.imread(image_file_path + ann[0] + ".jpg")
        mask_im = cv2.imread(mask_file_path + "mask." + ann[0] + ".png", 0)

        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.imshow('mask', mask_im * 255)
        cv2.waitKey(0)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        red_1_high = np.array((5, 255, 255), dtype = 'uint8')
        red_1_low = np.array((0, 50, 50), dtype = 'uint8')
        red_2_low = (175, 50, 50)
        red_2_high = (180, 255, 255)

        mask_hue_red_1 = cv2.inRange(hsv_image, red_1_low, red_1_high)
        mask_hue_red_2 = cv2.inRange(hsv_image, red_2_low, red_2_high)
        combined_red_mask = cv2.bitwise_or(mask_hue_red_1, mask_hue_red_2)

        cv2.imshow("mask_red", mask_hue_red_1)
        cv2.waitKey(0)
        cv2.imshow("mask_red", mask_hue_red_2)
        cv2.waitKey(0)
        cv2.imshow("mask_red", combined_red_mask)
        cv2.waitKey(0)

def blue_test(gt, mask_file_path, image_file_path):
    for ann in gt:
        image = cv2.imread(image_file_path + ann[0] + ".jpg")
        mask_im = cv2.imread(mask_file_path + "mask." + ann[0] + ".png", 0)

        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.imshow('mask', mask_im * 255)
        cv2.waitKey(0)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        blue_low = np.array((105, 30, 30), dtype='uint8')
        blue_high = np.array((135, 255, 255), dtype='uint8')


        mask_hue_blue = cv2.inRange(hsv_image, blue_low, blue_high)

        cv2.imshow("mask_blue", mask_hue_blue)
        cv2.waitKey(0)

def color_segmentation_hsv(gt, mask_file_path, image_file_path, results_file_path):
    for ann in gt:

        image = cv2.imread(image_file_path + ann[0] + ".jpg")
        mask_im = cv2.imread(mask_file_path + "mask." + ann[0] + ".png", 0)

        #cv2.imshow('image', image)
        #cv2.imshow('mask annotated', mask_im * 255)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        blue_low = np.array((105, 30, 30), dtype='uint8')
        blue_high = np.array((135, 255, 255), dtype='uint8')

        red_1_high = np.array((5, 255, 255), dtype='uint8')
        red_1_low = np.array((0, 50, 50), dtype='uint8')
        red_2_low = (175, 50, 50)
        red_2_high = (180, 255, 255)

        mask_hue_red_1 = cv2.inRange(hsv_image, red_1_low, red_1_high)
        mask_hue_red_2 = cv2.inRange(hsv_image, red_2_low, red_2_high)

        combined_red_mask = cv2.bitwise_or(mask_hue_red_1, mask_hue_red_2)
        mask_hue_blue = cv2.inRange(hsv_image, blue_low, blue_high)

        final_mask = cv2.bitwise_or(combined_red_mask, mask_hue_blue)

        mask_name = results_file_path + "mask." + ann[0] + ".png"
        cv2.imwrite(mask_name, final_mask)

        #cv2.imshow("method mask", final_mask)
        #cv2.waitKey(0)




def color_analysis(gt, mask_file_path, image_file_path):
    """
    This function extracts information about the color.
    :param gt: gt annotations: [image id, [tly, tlx, bry, brx], sign type, aspect ratio]
    :param mask_file_path: path to the masks' folder
    :param image_file_path: path to the train' folder
    :return:
    """
    histograms_by_signal = {}
    for ann in gt:
        image = cv2.imread(image_file_path + ann[0] + ".jpg")
        mask_im = cv2.imread(mask_file_path + "mask." + ann[0] + ".png", 0)

        """colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], mask_im, [256], [0, 256])
            plt.plot(hist, color=color)
        plt.show()"""

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        """cv2.imshow('image H', hsv_image[:, :, 0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('image S', hsv_image[:, :, 1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('image V', hsv_image[:, :, 2])
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""

        histH = cv2.calcHist([hsv_image], [0], mask_im, [256], [0, 256])
        histS = cv2.calcHist([hsv_image], [1], mask_im, [256], [0, 256])
        histV = cv2.calcHist([hsv_image], [2], mask_im, [256], [0, 256])
        sign_type = ann[2]
        if sign_type in histograms_by_signal:
            histograms_by_signal[sign_type].append((histH, histS, histV))
        else:
            histograms_by_signal[sign_type] = [(histH, histS, histV)]

    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
    for sign_type in histograms_by_signal:
        for i, hist in enumerate(histograms_by_signal[sign_type]):
            plt.subplot(3, 1, 1)
            plt.plot(hist[0], color=colors[i%len(colors)])
            plt.title("Hue  "+sign_type + ": " + parse_sign_type(sign_type))
            plt.subplot(3, 1, 2)
            plt.plot(hist[1], color=colors[i % len(colors)])
            plt.title("Saturation  " + sign_type + ": " + parse_sign_type(sign_type))
            plt.subplot(3, 1, 3)
            plt.plot(hist[2], color=colors[i % len(colors)])
            plt.title("Value  " + sign_type + ": " + parse_sign_type(sign_type))

        plt.title(sign_type+": "+parse_sign_type(sign_type))
        plt.show()


def main():
    gt_file_path = "./train/gt/"
    image_file_path = "./train/"
    mask_file_path = "./train/mask/"
    results_file_path = "./results/"

    gt = read_gt(gt_file_path)
    color_analysis(gt, mask_file_path=mask_file_path, image_file_path=image_file_path)
    red_test(gt, mask_file_path=mask_file_path, image_file_path=image_file_path)
    # color_segmentation_hsv(gt, mask_file_path=mask_file_path, image_file_path=image_file_path, results_file_path=results_file_path)


if __name__ == "__main__":
    main()