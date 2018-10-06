#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from skimage import color
from traffic_sign_model import Traffic_sign_model

traffic_sign_model = Traffic_sign_model()

def candidate_generation_pixel_hsvclosing(image):
    return traffic_sign_model.pixel_method(image)

def candidate_generation_pixel_normrgb(im):
    # convert input image to the normRGB color space

    normrgb_im = np.zeros(im.shape)
    eps_val = 0.00001
    norm_factor_matrix = im[:,:,0] + im[:,:,1] + im[:,:,2] + eps_val

    normrgb_im[:,:,0] = im[:,:,0] / norm_factor_matrix
    normrgb_im[:,:,1] = im[:,:,1] / norm_factor_matrix
    normrgb_im[:,:,2] = im[:,:,2] / norm_factor_matrix
    
    # Develop your method here:
    # Example:
    pixel_candidates = normrgb_im[:,:,1]>100;

    return pixel_candidates
 
def candidate_generation_pixel_hsv(image):
    # convert input image to HSV color space
    # hsv_im = color.rgb2hsv(im)
    
    # Develop your method here:
    # Example:
    # pixel_candidates = hsv_im[:,:,1] > 0.4;

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    blue_low = np.array((105, 30, 30), dtype='uint8')
    blue_high = np.array((135, 255, 255), dtype='uint8')

    red_1_low = np.array((0, 35, 50), dtype='uint8')
    red_1_high = np.array((5, 255, 255), dtype='uint8')
    red_2_low = (175, 35, 50)
    red_2_high = (180, 255, 255)

    mask_hue_red_1 = cv2.inRange(hsv_image, red_1_low, red_1_high)
    mask_hue_red_2 = cv2.inRange(hsv_image, red_2_low, red_2_high)

    combined_red_mask = cv2.bitwise_or(mask_hue_red_1, mask_hue_red_2)

    mask_hue_blue = cv2.inRange(hsv_image, blue_low, blue_high)
    pixel_candidates = cv2.bitwise_or(combined_red_mask, mask_hue_blue)

    # morph test:
    kernel = np.ones((5, 5), np.uint8)
    pixel_candidates = cv2.morphologyEx(pixel_candidates, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((10, 10), np.uint8)
    pixel_candidates = cv2.morphologyEx(pixel_candidates, cv2.MORPH_CLOSE, kernel)

    image, contours, hierarchy = cv2.findContours(pixel_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_aspect_ratio = 1.419828704905269 * 1.15
    min_aspect_ratio = 0.442105263157894 * 0.85
    max_area = 55919.045 * 1.10
    min_area = 909.7550000000047 * 0.75

    for contour in contours:
        xcnts = np.vstack(contour.reshape(-1, 2))
        x_min = min(xcnts[:, 0])
        x_max = max(xcnts[:, 0])
        y_min = min(xcnts[:, 1])
        y_max = max(xcnts[:, 1])

        width = y_max - y_min
        height = x_max - x_min

        if max_aspect_ratio > height/width > min_aspect_ratio and max_area > width*height > min_area:
            cv2.fillPoly(image, pts=[contour], color=255)
        else:
            for x in range(y_min-1, y_max+1):
                for y in range(x_min-1, x_max+1):
                    image[x, y] = 0

    return image  # change to pixel candidates
 


# Create your own candidate_generation_pixel_xxx functions for other color spaces/methods
# Add them to the switcher dictionary in the switch_color_space() function
# These functions should take an image as input and output the pixel_candidates mask image
 
def switch_color_space(im, color_space):
    switcher = {
        'normrgb': candidate_generation_pixel_normrgb,
        'hsv'    : candidate_generation_pixel_hsv,
        'hsvclosing': candidate_generation_pixel_hsvclosing
        #'lab'    : candidate_generation_pixel_lab,
    }
    # Get the function from switcher dictionary
    func = switcher.get(color_space, lambda: "Invalid color space")

    # Execute the function
    pixel_candidates =  func(im)

    return pixel_candidates


def candidate_generation_pixel(im, color_space):

    pixel_candidates = switch_color_space(im, color_space)

    return pixel_candidates

    
if __name__ == '__main__':
    pixel_candidates1 = candidate_generation_pixel(im, 'normrgb')
    pixel_candidates2 = candidate_generation_pixel(im, 'hsv')

    
