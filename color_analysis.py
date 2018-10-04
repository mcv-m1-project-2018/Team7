from db_analysis import read_gt
import cv2
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
        """cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('mask', mask_im * 255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('image', image[:,:,0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('image', image[:, :, 1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('image', image[:, :, 2])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], mask_im, [256], [0, 256])
            plt.plot(hist, color=color)
        plt.show()
        """
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

        hist = cv2.calcHist([hsv_image], [0], mask_im, [256], [0, 256])
        sign_type = ann[2]
        if sign_type in histograms_by_signal:
            histograms_by_signal[sign_type].append(hist)
        else:
            histograms_by_signal[sign_type] = [hist]

    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
    for sign_type in histograms_by_signal:
        print(sign_type+": "+parse_sign_type(sign_type))
        for i, hist in enumerate(histograms_by_signal[sign_type]):
            plt.plot(hist, color=colors[i%len(colors)])
        plt.show()


def main():
    gt_file_path = "./train/gt/"
    image_file_path = "./train/"
    mask_file_path = "./train/mask/"

    gt = read_gt(gt_file_path)
    color_analysis(gt, mask_file_path=mask_file_path, image_file_path=image_file_path)


if __name__ == "__main__":
    main()