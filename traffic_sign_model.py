import argparse
import numpy as np
import cv2

from data import Data_handler
from data_analysis import Data_analysis
from traffic_signs import traffic_sign_detection as detection
import matplotlib.pyplot as plt
import os
import time

class Traffic_sign_model():
    def __init__(self):
        self.parameters= {  # [optimal_value, start_range, end_range]
            'blue_low_h': [104, 90, 140],
            'blue_low_s': [49, 20, 255],
            'blue_low_v': [31, 20, 255],
            'blue_high_h': [136, 90, 140],
            'blue_high_s': [254, 20, 255],
            'blue_high_v': [239, 20, 255],
            'red1_low_h': [0, 0, 25],
            'red1_low_s': [67, 20, 255],
            'red1_low_v': [55, 20, 255],
            'red1_high_h': [7, 0, 25],
            'red1_high_s': [255, 20, 255],
            'red1_high_v': [255, 20, 255],
            'red2_low_h': [178, 165, 180],
            'red2_low_s': [66, 20, 255],
            'red2_low_v': [56, 20, 255],
            'red2_high_h': [180, 165, 180],
            'red2_high_s': [255, 20, 255],
            'red2_high_v': [249, 20, 255],
        }
        self.MAX_RANGE = 40 #max range to search for optimal parameter

    def color_segmentation(self, im):
        """
        Color segmentation of red and blue regions
        :param im: BGR image
        :return: mask with the color segmentation
        """

        hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        blue_low = np.array(
            (self.parameters['blue_low_h'], self.parameters['blue_low_s'], self.parameters['blue_low_v']),
            dtype='uint8')
        blue_high = np.array(
            (self.parameters['blue_high_h'], self.parameters['blue_high_s'], self.parameters['blue_high_v']),
            dtype='uint8')

        red_1_low = np.array(
            (self.parameters['red1_low_h'], self.parameters['red1_low_s'], self.parameters['red1_low_v']),
            dtype='uint8')
        red_1_high = np.array(
            (self.parameters['red1_high_h'], self.parameters['red1_high_s'], self.parameters['red1_high_v']),
            dtype='uint8')

        red_2_low = np.array(
            (self.parameters['red2_low_h'], self.parameters['red2_low_s'], self.parameters['red2_low_v']),
            dtype='uint8')
        red_2_high = np.array(
            (self.parameters['red2_high_h'], self.parameters['red2_high_s'], self.parameters['red2_high_v']),
            dtype='uint8')

        mask_hue_red_1 = cv2.inRange(hsv_image, red_1_low, red_1_high)
        mask_hue_red_2 = cv2.inRange(hsv_image, red_2_low, red_2_high)

        combined_red_mask = cv2.bitwise_or(mask_hue_red_1, mask_hue_red_2)
        mask_hue_blue = cv2.inRange(hsv_image, blue_low, blue_high)
        final_mask = cv2.bitwise_or(combined_red_mask, mask_hue_blue)

        return final_mask

    def pixel_method(self, im):
        """
        Color segmentation of red and blue regions and morphological transformations
        :param im: BGR image
        :return: mask with the pixel candidates
        """

        color_segmentation_mask = self.color_segmentation(im)

        pixel_candidates = self.morph_transformation(color_segmentation_mask)

        return pixel_candidates

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

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
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

    def template_matching(self, im, pixel_candidates, threshold=.6, show=False):
        """
        Matches the templates found in ./data/templates with the candidate regions of the image. Keeps those with
        a higher score than the threshold.
        :param im: the image (in bgr)
        :param pixel_candidates: the mask
        :param threshold: threshold score for the different areas.
        :param show: if true shows the regions and their scores
        :return: mask, window_candidates
        """
        final_mask = pixel_candidates
        window_candidates = []

        # read the templates
        templates = []
        template_filenames = os.listdir("./data/templates/")
        for filename in template_filenames:
            template = cv2.imread("./data/templates/" + filename)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY).astype(np.float32)
            templates.append(template)

        _, contours, _ = cv2.findContours(pixel_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if show:
            import matplotlib.pyplot as plt
            import matplotlib.patches as pat
            fig, ax = plt.subplots(1)
            ax.imshow(im, cmap="gray")

        # process every region found
        for contour in contours:
            xcnts = np.vstack(contour.reshape(-1, 2))
            x_min = min(xcnts[:, 0])
            x_max = max(xcnts[:, 0])
            y_min = min(xcnts[:, 1])
            y_max = max(xcnts[:, 1])
            padding = 30
            # cops the interest region + a padding
            region = im[max(0, y_min - padding):min(im.shape[0], y_max + padding),
                     max(0, x_min - padding):min(im.shape[1], x_max + padding)]
            region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY).astype(np.float32)

            dsize = min(region.shape)
            max_score = 0
            scalars = [1]  # this is to give different scales to the template, not sure if we should use it

            # print((dsize, dsize), ", ", region.shape)

            for template in templates:
                for scalar in scalars:
                    dsize_scaled = int(dsize * scalar)
                    template_g = cv2.resize(template, dsize=(dsize_scaled, dsize_scaled), interpolation=cv2.INTER_CUBIC)
                    res = cv2.matchTemplate(region, template_g, cv2.TM_CCOEFF_NORMED)
                    max_temp_score = np.max(res)

                    if max_temp_score > max_score:
                        max_score = max_temp_score
            if show:
                rec = pat.Rectangle((x_min - padding, y_min - padding), (x_max + padding) - (x_min - padding),
                                    (y_max + padding) - (y_min - padding)
                                    , linewidth=1, edgecolor='r', facecolor='none')
                plt.text(x_min, y_min, str(max_score), color="red", size=15)
                ax.add_patch(rec)

            if max_score < threshold:  # delete the region if the score is too low
                cv2.fillPoly(pixel_candidates, pts=[contour], color=0)
        if show:
            plt.show()

        # calculates the windows for all the regions
        _, contours, _ = cv2.findContours(pixel_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            xcnts = np.vstack(contour.reshape(-1, 2))
            x_min = min(xcnts[:, 0])
            x_max = max(xcnts[:, 0])
            y_min = min(xcnts[:, 1])
            y_max = max(xcnts[:, 1])
            window_candidates.append([y_min, x_min, y_max, x_max])

        return final_mask, window_candidates

    def window_method(self, im, pixel_candidates):
        pass
        # return window_candidates

    def tuning_f1(self,train_set,valid_set):
        pass

####################################################### PARAMETER OPTIMIZATION ##################################################
    def save_progress(self, parameter, t1, current_max_value, current_precision, current_sensitivity):
        with open('optimization_parameters.log', "a") as f:
            f.write("-------------------------------- " + parameter + "\n")
            f.write("total time: " + str((time.time() - t1) / 60) + " min\n")
            f.write(str(self.parameters) + "\n")
            f.write("Score: " + str(current_max_value) + "\n")
            f.write("Precision: " + str(current_precision) + "\n")
            f.write("Sensitivity: " + str(current_sensitivity) + "\n")
        print("-------------------------------- " + parameter)
        print("total time: " + str((time.time() - t1) / 60) + " min\n")
        print(str(self.parameters))
        print("Score: " + str(current_max_value))
        print("Precision: " + str(current_precision))
        print("Sensitivity: " + str(current_sensitivity))

    def score(precision, sensitivity): #Score function to be optimised
        return 0.7 * sensitivity + 0.3 * precision

    def optimize_parameters(self, train_set):

        with open('optimization_parameters.log', "a") as f:
            f.write("********************* New Execution ***************************\n")
        t1 = time.time()

        [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity] = self.evaluate_parameters() #TO DO

        current_max_value = self.score(precision=pixel_precision, sensitivity=pixel_sensitivity)
        currrent_precision = pixel_precision
        current_sensitivity = pixel_sensitivity
        last_current_max_value = 0

        while (current_max_value - last_current_max_value > 0.001): #Tune to redefine when we consider the parameters have converged
            last_current_max_value = current_max_value
            for parameter in self.parameters:

                current_parameter = self.parameters[parameter][0]
                start_range = max(self.parameters[parameter][1], current_parameter - MAX_RANGE // 2)
                end_range = min(self.parameters[parameter][2], current_parameter + MAX_RANGE // 2)

                for p in range(start_range, end_range):
                    if (p != current_parameter):
                        print(p)
                        self.parameters[parameter][0] = p

                        [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity] = self.evaluate_parameters() #TO DO

                        value = self.score(precision=pixel_precision, sensitivity=pixel_sensitivity)
                        print(value)
                        if (value > current_max_value):
                            current_parameter = p
                            current_max_value = value
                            current_precision = pixel_precision
                            current_sensitivity = pixel_sensitivity

                self.parameters[parameter][0] = current_parameter
                self.save_progress(parameter=parameter,
                              t1=t1,
                              current_max_value=current_max_value,
                              current_precision=current_precision,
                              current_sensitivity=current_sensitivity)


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
                                         model.pixel_method, args.windowMethod)

    print(pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity, window_precision, window_accuracy)

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
