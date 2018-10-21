
from traffic_sign_model import Traffic_sign_model
import matplotlib.pyplot as plt

import argparse
import numpy as np
import cv2

from data import Data_handler
from data_analysis import Data_analysis
from traffic_signs import traffic_sign_detection as detection
from traffic_signs.evaluation.bbox_iou import bbox_iou
import matplotlib.pyplot as plt
import os
import time


class method_2(Traffic_sign_model):
    def __init__(self):
        super().__init__()
        self.pixel_method_name = 'hsv_seg_morph'
        self.window_method_name = 'templates_sliding'
        self.parameters = {  # [optimal_value, start_range, end_range]
            'blue_low_h': [104, 90, 140],
            'blue_low_s': [49, 20, 255],
            'blue_low_v': [31, 20, 255],

            'blue_high_h': [136, 90, 140],
            'blue_high_s': [254, 20, 255],
            'blue_high_v': [239, 20, 255],

            'red1_low_h': [0, 0, 25],
            'red1_low_s': [67, 20, 255],
            'red1_low_v': [55, 20, 255],

            'red1_high_h': [10, 0, 25],
            'red1_high_s': [255, 20, 255],
            'red1_high_v': [255, 20, 255],

            'red2_low_h': [170, 165, 180],
            'red2_low_s': [66, 20, 255],
            'red2_low_v': [56, 20, 255],

            'red2_high_h': [180, 165, 180],
            'red2_high_s': [255, 20, 255],
            'red2_high_v': [249, 20, 255],
            'threshold_template_matching': [0.5, 0, 1]
        }

    def pixel_method(self, im):
        """
        Color segmentation of red and blue regions and morphological transformations
        :param im: BGR image
        :return: mask with the pixel candidates
        """

        color_segmentation_mask = self.color_segmentation(im)
        pixel_candidates = self.morph_transformation(color_segmentation_mask)
        pixel_candidates = self.ccl_generation_filtering(pixel_candidates)

        return pixel_candidates

    def window_method(self, im, pixel_candidates):
        # Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
        # cv2.imshow('o',im)
        # cv2.imshow('pixel_candidates',pixel_candidates)

        # window_candidates = self.get_ccl_bbox(pixel_candidates)

        final_mask, window_candidates = self.template_matching(im, pixel_candidates, show=False)

        # for w in window_candidates:
        #    cv2.rectangle(im,(w[1],w[0]),(w[3],w[2]),(0,255,0),3)
        # cv2.imshow('boxes',im)
        # cv2.waitKey()

        return window_candidates

    def template_matching(self, im, pixel_candidates, show):
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
        im_h, im_w, _ = im.shape
        # read the templates
        templates = []
        templates_mask = []
        template_filenames = os.listdir("./data/templates/")

        for filename in template_filenames:
            template = cv2.imread("./data/templates/" + filename)
            template_mask = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            _, template_mask = cv2.threshold(template_mask, 5, 255, cv2.THRESH_BINARY)
            template_mask = cv2.cvtColor(template_mask, cv2.COLOR_GRAY2BGR)
            templates.append(template)
            templates_mask.append(template_mask)

        _, contours, _ = cv2.findContours(pixel_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # process every region found
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            min_score = 100000

            if x < im_w and y < im_h and (x + width) < im_w and (y + height) < im_h:
                region = np.copy(im[y:(y + height), x:(x + width), :])
                region_shape = region.shape
                region_resized = cv2.resize(region, (100, 100))

                for template, template_mask in zip(templates, templates_mask):
                    # = cv2.bitwise_and(region_resized, templates_mask)
                    # region_masked = region_resized.copy()
                    # region_masked[template_mask] = 0
                    # region_resized[template_mask] = 0
                    # print(template_mask.shape  )
                    # print(region_resized.shape )
                    # template_mask = cv2.resize(region, (region_resized.shape[0], region_resized.shape[1]))
                    region_masked = cv2.bitwise_and(region_resized, template_mask)
                    res = cv2.matchTemplate(region_masked, template, cv2.TM_SQDIFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                    if min_val < min_score:
                        min_score = min_val
                        top_left = (min_loc[0] + x, min_loc[1] + y)
                    if show:
                        cv2.imshow('pixel_candidates', pixel_candidates)
                        cv2.imshow('template', template)
                        cv2.imshow('w', region_masked)
                        cv2.imshow('template_mask', template_mask)
                        #print(min_val)
                        cv2.waitKey()

                bottom_right = (top_left[0] + region_shape[1], top_left[1] + region_shape[0])

            if min_score < self.parameters['threshold_template_matching']:
                window_candidates.append([top_left[1], top_left[0], bottom_right[1], bottom_right[0]])
                # window_candidates.append([y,x,y+height,x+width])
                if show:
                    w = window_candidates[-1]
                    cv2.rectangle(im, (w[1], w[0]), (w[3], w[2]), (0, 255, 0), 3)
                    cv2.imshow('im', im)
            #print(min_score)

        return final_mask, window_candidates



#model.evaluate(split='train', output_dir='test_results/')
    def PRC(self, parameter_name, num_values):
        precision = []
        recall = []
        r = self.parameters[parameter_name][1:]
        for i in range(num_values):
            value = r[0] + (r[1]-r[0])*i/(num_values-1)
            print('Threshold: '+str(value))
            self.parameters[parameter_name] = value
            eval = self.evaluate(split='train', output_dir='test_results/')
            precision.append(eval['window_precision'])
            recall.append(eval['window_sensitivity'])

        print(precision)
        print(recall)
        fig = plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision - Recall Curve')
        plt.show()
        #plt.savefig('PRC.png')



model = method_2()
model.PRC(parameter_name='threshold_template_matching', num_values=11)

"""fig = plt.plot([1,0.8,0.5,0], [0,0.3,0.6,1])
#plt.axis([0, 1, 0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision - Recall Curve')

plt.show()
#plt.savefig('PRC.png')"""