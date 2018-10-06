import argparse
import numpy as np
import cv2

from data import Data_handler
from data_analysis import Data_analysis

"""
This is the main file. I've created a "main" function where the data structures are initialized just as an
example.   
"""

class Traffic_sign_model():
    # hyperparameters
    def pixel_method(self, im, blue_low_hsv= (90 ,50,50), blue_high_hsv= (140,255,255), 
                               red1_low_hsv= (0  ,50,50), red1_high_hsv= (25 ,255,255),
                               red2_low_hsv= (165,50,50), red2_high_hsv= (180,255,255)):

        hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        blue_low   = np.array(blue_low_hsv, dtype='uint8')
        blue_high  = np.array(blue_high_hsv,    dtype='uint8')
        
        red_1_low  = np.array(red1_low_hsv, dtype='uint8')
        red_1_high = np.array(red1_high_hsv, dtype='uint8')

        red_2_low  = np.array(red2_low_hsv, dtype='uint8')
        red_2_high = np.array(red2_high_hsv, dtype='uint8')

        mask_hue_red_1 = cv2.inRange(hsv_image, red_1_low, red_1_high)
        mask_hue_red_2 = cv2.inRange(hsv_image, red_2_low, red_2_high)

        combined_red_mask = cv2.bitwise_or(mask_hue_red_1, mask_hue_red_2)
        mask_hue_blue     = cv2.inRange(hsv_image, blue_low, blue_high)
        final_mask        = cv2.bitwise_or(combined_red_mask, mask_hue_blue)        

        return final_mask

    def window_method(self, im, pixel_candidates):
        pass
        # return window_candidates

    def tuning_f1(self,train_set,valid_set):
        pass


def main():
    data_hdlr = data_handler(train_dir=args.images_dir)

    sign_count, max_area, min_area, filling_ratios, max_aspect_ratio, min_aspect_ratio = data_analysis.shape_analysis\
        (data_hdlr.train_set)

    # data_analysis.color_analysis(data_hdlr.train_set) # works but returns nothing

    for key in filling_ratios.keys():
        print(key + ": " + str(filling_ratios[key]))

    print("sign_count: ", sign_count, "\n", "max_area: ", max_area, "\n", "min_area: ", min_area,
          "\n", "max_aspect_ratio: ", max_aspect_ratio, "\n", "min_aspect_ratio: ", min_aspect_ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-images_dir', type=str, default="./train/", help='Directory with input images and '
                                                                          'annotations')
    parser.add_argument('-output_dir', type=str, default="./results/", help='Directory where to store output masks, '
                                                                            'etc.  For instance ~/m1-results/week1/test'
                                                                            '')
    parser.add_argument('-pixelMethod', type=str, default="hsv", help='Colour space used during the segmentation'
                                                                      '(either hsv or normrgb)')
    parser.add_argument('-windowMethod', type=str, default="ok", help='this parameter is a mystery for us')
    args = parser.parse_args()
    main()
