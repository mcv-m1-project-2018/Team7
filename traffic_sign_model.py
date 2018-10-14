import argparse
import numpy as np
import cv2

from data import Data_handler
from data_analysis import Data_analysis
from traffic_signs import traffic_sign_detection as detection


class Traffic_sign_model():
    def __init__(self):
        self.blue_low_hsv = (105, 30, 30)
        self.blue_high_hsv = (135, 255, 255)
        self.red1_low_hsv = (0, 50, 50)
        self.red1_high_hsv = (8, 255, 255)
        self.red2_low_hsv = (177, 50, 50)
        self.red2_high_hsv = (181, 255, 255)


    def pixel_method(self, im):

        hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        blue_low   = np.array(self.blue_low_hsv, dtype='uint8')
        blue_high  = np.array(self.blue_high_hsv, dtype='uint8')
        
        red_1_low  = np.array(self.red1_low_hsv, dtype='uint8')
        red_1_high = np.array(self.red1_high_hsv, dtype='uint8')

        red_2_low  = np.array(self.red2_low_hsv, dtype='uint8')
        red_2_high = np.array(self.red2_high_hsv, dtype='uint8')

        mask_hue_red_1 = cv2.inRange(hsv_image, red_1_low, red_1_high)
        mask_hue_red_2 = cv2.inRange(hsv_image, red_2_low, red_2_high)

        combined_red_mask = cv2.bitwise_or(mask_hue_red_1, mask_hue_red_2)
        mask_hue_blue     = cv2.inRange(hsv_image, blue_low, blue_high)
        final_mask        = cv2.bitwise_or(combined_red_mask, mask_hue_blue)

        final_mask = self.morph_transformation(final_mask)

        return final_mask


    def morph_transformation(self, pixel_candidates):
        """
        Performs morphological operations over the masks.

        :param pixel_candidates: the pixels of the image
        :return: more pixels
        """

        # kernel = np.ones((5, 5), np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        pixel_candidates = cv2.morphologyEx(pixel_candidates, cv2.MORPH_OPEN, kernel)

        # kernel = np.ones((10, 10), np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        pixel_candidates = cv2.morphologyEx(pixel_candidates, cv2.MORPH_CLOSE, kernel)

        # find all contours (segmented areas) of the mask to delete those that are not consistent with the train split
        # analysis

        max_aspect_ratio = 1.419828704905269 * 1.25
        min_aspect_ratio = 0.5513618362563639 * 0.75
        max_area = 55919.045 * 1.15
        min_area = 909.7550000000047 * 0.75

        image, contours, hierarchy = cv2.findContours(pixel_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            # max and min coordinates of the segmented area
            xcnts = np.vstack(contour.reshape(-1, 2))
            x_min = min(xcnts[:, 0])
            x_max = max(xcnts[:, 0])
            y_min = min(xcnts[:, 1])
            y_max = max(xcnts[:, 1])

            width = y_max - y_min
            height = x_max - x_min

            # check if the aspect ratio and area are bigger or smaller than the ground truth. If it is consistent with
            # the ground truth, we try to fill it (some signs are not fully segmented because they contain white or other
            # colors) with cv2.fillPoly.
            if max_aspect_ratio > height/width > min_aspect_ratio and max_area > width*height > min_area:
                cv2.fillPoly(pixel_candidates, pts=[contour], color=255)
            else:
                cv2.fillPoly(pixel_candidates, pts=[contour], color=0)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        pixel_candidates = cv2.dilate(pixel_candidates, kernel, iterations=1)

        image, contours, hierarchy = cv2.findContours(pixel_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            xcnts = np.vstack(contour.reshape(-1, 2))
            x_min = min(xcnts[:, 0])
            x_max = max(xcnts[:, 0])
            y_min = min(xcnts[:, 1])
            y_max = max(xcnts[:, 1])

            width = y_max - y_min
            height = x_max - x_min

            if max_aspect_ratio > height/width > min_aspect_ratio and max_area > width*height > min_area:
                cv2.fillPoly(pixel_candidates, pts=[contour], color=255)
            else:
                cv2.fillPoly(pixel_candidates, pts=[contour], color=0)


        return pixel_candidates


    def window_method(self, im, pixel_candidates):
        final_mask = pixel_candidates
        # window_candidates = [[0,0,1000,1000]]
        window_candidates = []


        return final_mask, window_candidates



    def tuning_f1(self,train_set,valid_set):
        pass


def main(args):
    """
    The main function initializes the database with all the annotation from the ground truth. The database is contained
    in the "data_hdlr" object (data.py). This object contains the val, train and test splits.
    To analyze the shape and other attributes of the annotations from the train split, we call the "shape_analysis"
    method from the "Data_analysis" class in the "data_analysis.py" file. The results of this analysis are printed
    in the terminal.
    Finally, we perform the segmentation over the validation split and we the metrics are also displayed through the
    terminal. The results are saved in ./results/segmentation-method_window-method_split. For the test split we only
    store the generated masks.
    :param args: arguments
    :return:
    """

    print("reading the data...")
    data_hdlr = Data_handler(train_dir=args.images_dir)
    data_hdlr.read_all()

    # print("analyzing the train split...\n")
    # sign_count, max_area, min_area, filling_ratios, max_aspect_ratio, min_aspect_ratio = Data_analysis.shape_analysis\
    #     (data_hdlr.train_set)

    # print("Creating templates...\n")
    # Data_analysis.create_templates(data_hdlr.train_set)

    model = Traffic_sign_model()
    #
    # for key in filling_ratios.keys():
    #     print(key + ": " + str(filling_ratios[key]))
    #
    # print("sign_count: ", sign_count, "\n", "max_area: ", max_area, "\n", "min_area: ", min_area,
    #       "\n", "max_aspect_ratio: ", max_aspect_ratio, "\n", "min_aspect_ratio: ", min_aspect_ratio)

    print("\nprocessing the val split...\n")
    pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity, window_precision, window_accuracy = \
        detection.traffic_sign_detection("val", args.images_dir, data_hdlr.valid_set, args.output_dir, 'hsvClosing',
                                         model.pixel_method, model.window_method)



    print(pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity, window_precision, window_accuracy)
    #
    # print("\nprocessing the test split...")
    #
    # detection.traffic_sign_detection("test", args.test_dir, data_hdlr.test_set, args.output_dir, 'hsvClosing',
    #                                  model.pixel_method, args.windowMethod)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-images_dir', type=str, default="./train/", help='Directory with input images and '
                                                                          'annotations')
    parser.add_argument('-output_dir', type=str, default="./results/", help='Directory where to store output masks, '
                                                                            'etc.  For instance ~/m1-results/week1/test'
                                                                            '')
    parser.add_argument('-test_dir', type=str, default="./test/", help='Directory with the test split images')
    parser.add_argument('-pixelMethod', type=str, default="hsv", help='Colour space used during the segmentation'
                                                                      '(either hsv or normrgb)')
    parser.add_argument('-windowMethod', type=str, default="None", help='this parameter is a mystery for us')
    args = parser.parse_args()
    main(args)
