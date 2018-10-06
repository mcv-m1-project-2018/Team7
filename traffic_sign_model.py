import argparse

from data import data_handler
from data_analysis import data_analysis


"""
This is the main file. I've created a "main" function where the data structures are initialized just as an
example.   
"""


class traffic_sign_model():
    # hyperparameters
    def pixel_method(im):
        pass
        # return pixel_candidates

    def window_method(im, pixel_candidates):
        pass
        # return window_candidates

    @staticmethod
    def tuning_f1():
        pass


def main():
    data_hdlr = data_handler(train_dir=args.images_dir)

    sign_count, max_area, min_area, filling_ratios, max_aspect_ratio, min_aspect_ratio = data_analysis.shape_analysis \
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
