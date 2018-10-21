import argparse
import numpy as np
import cv2

from data import Data_handler
from data_analysis import Data_analysis
from traffic_signs import traffic_sign_detection as detection
from traffic_signs.evaluation.bbox_iou import bbox_iou
from traffic_sign_model import Traffic_sign_model
import matplotlib.pyplot as plt
import os
import time


class method_1(Traffic_sign_model):
    def __init__(self):        
        super().__init__()
        self.pixel_method_name  = '1_hsvMorphFiltering'
        self.window_method_name = '1_jsfkjdsk'
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

            'red1_high_h': [10, 0, 25],
            'red1_high_s': [255, 20, 255],
            'red1_high_v': [255, 20, 255],  


            'red2_low_h': [170, 165, 180],
            'red2_low_s': [66, 20, 255],
            'red2_low_v': [56, 20, 255],

            'red2_high_h': [180, 165, 180],
            'red2_high_s': [255, 20, 255],
            'red2_high_v': [249, 20, 255],
        }

    def pixel_method(self, im):
        """
        Color segmentation of red and blue regions and morphological transformations
        :param im: BGR image
        :return: mask with the pixel candidates
        """
        
        
        color_segmentation_mask = self.color_segmentation(im)
        pixel_candidates        = self.morph_transformation(color_segmentation_mask)
        pixel_candidates        = self.ccl_generation_filtering(pixel_candidates)
        
        
        
        return pixel_candidates

    def window_method(self, im, pixel_candidates):
    # Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
        #cv2.imshow('o',im)
        cv2.imshow('pixel_candidates',pixel_candidates)
        cv2.waitKey(1)
        #window_candidates = self.get_ccl_bbox(pixel_candidates)

        final_mask, window_candidates, score_candidates = self.template_matching( im, pixel_candidates, threshold=.4, show=False)
        new_window_candidates = self.remove_overlapped( window_candidates, score_candidates )
        for w in new_window_candidates:
            cv2.rectangle(im,(w[1],w[0]),(w[3],w[2]),(0,255,0),3)
        cv2.imshow('im',im)
        cv2.waitKey(1)

        #for w in window_candidates:
        #    cv2.rectangle(im,(w[1],w[0]),(w[3],w[2]),(0,255,0),3)
        #cv2.imshow('boxes',im)
        #cv2.waitKey()

        return new_window_candidates    

    def template_matching(self, im, pixel_candidates, threshold, show):
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
        templates        = []
        templates_mask   = []
        score_candidates = []
        template_filenames = os.listdir("./data/templates/")

        for filename in template_filenames:
            template = cv2.imread("./data/templates/" + filename)
            template_mask = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            _,template_mask = cv2.threshold(template_mask,5,255,cv2.THRESH_BINARY)
            template_mask = cv2.cvtColor(template_mask, cv2.COLOR_GRAY2BGR)
            templates.append(template)
            templates_mask.append(template_mask)

        _, contours, _ = cv2.findContours(pixel_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # process every region found
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            """
            x = int(min(max(x - width/2, 0),  im_w-1))
            y = int(min(max(y - height/2, 0), im_h-1))
            width  = max(width*2,100)
            height = max(height*2,100)
            """
            min_score = 100000
            
            if x<im_w and y<im_h and (x+width) < im_w and (y+height)<im_h:
                region = np.copy(im[y:(y+height),x:(x+width), :])
                #region_shape    = region.shape
                region_resized  = cv2.resize(region, (100, 100))
                sliding_windows = self.sliding_window(region)
                
                for template, template_mask in zip(templates,templates_mask):
                    for w in sliding_windows:
                        region = np.copy(im[w[1]:(w[1]+w[3]),w[0]:(w[0]+w[2]), :])
                        #region_resized  = cv2.resize(region, (100, 100))
                        #= cv2.bitwise_and(region_resized, templates_mask)
                        #region_masked = region_resized.copy()
                        #region_masked[template_mask] = 0
                        #region_resized[template_mask] = 0
                        #print(template_mask.shape  )
                        #print(region_resized.shape )
                        template_mask = cv2.resize(template_mask, (region_resized.shape[1], region_resized.shape[0]))
                        region_masked = cv2.bitwise_and(region_resized, template_mask)
                        res = cv2.matchTemplate(region_masked, template, cv2.TM_SQDIFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    

                        if min_val < min_score:
                            min_score = min_val
                            top_left = (min_loc[0]+x+w[0],min_loc[1]+y+w[1])
                        if show:
                            cv2.imshow('pixel_candidates',pixel_candidates)
                            cv2.imshow('template',template)
                            cv2.imshow('w',region_masked)
                            cv2.imshow('template_mask',template_mask)
                            print(min_val)
                            cv2.waitKey()

                        bottom_right = (top_left[0] + w[2], top_left[1] + w[3])

            
                        if min_score<threshold:
                            window_candidates.append([top_left[1],top_left[0], bottom_right[1],bottom_right[0]])
                            score_candidates.append(min_score)
                            #window_candidates.append([y,x,y+height,x+width])
                            #if show:
                        print(min_score)

        return final_mask, window_candidates, score_candidates


    def sliding_window(self, search_region):
        im_height,im_width          = search_region.shape[:2]
        slices       = []
        width_range  = [int(im_width), int(im_width/2), int(im_width/3), int(im_width/4)]
        height_range = [int(im_height), int(im_height/2), int(im_height/3), int(im_height/4)]
        x_step       = im_width#max(int(im_width/2),2)
        y_step       = im_height#max(int(im_height/2),2)
        for w in width_range:
            for h in height_range:
                for x in range(0, im_width-w+1, x_step):
                    for y in range(0, im_height-h+1, y_step):
                        slices.append([x,y,w,h])

        #slices = [[0,0,im_width,im_height]]
        return slices