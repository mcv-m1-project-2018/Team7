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
		#cv2.imshow('pixel_candidates',pixel_candidates)

		window_candidates = self.get_ccl_bbox(pixel_candidates)
		#for w in window_candidates:
		#	cv2.rectangle(im,(w[1],w[0]),(w[3],w[2]),(0,255,0),3)
		#cv2.imshow('boxes',im)
		#cv2.waitKey()
		#final_mask, window_candidates = self.template_matching( im, pixel_candidates, threshold=.2, show=False)

		return window_candidates	