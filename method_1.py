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

	def pixel_method(self, im):
		"""
		Color segmentation of red and blue regions and morphological transformations
		:param im: BGR image
		:return: mask with the pixel candidates
		"""
		#cv2.imshow('o',im)
		color_segmentation_mask = self.color_segmentation(im)
		#cv2.imshow('after seg',color_segmentation_mask)

		pixel_candidates        = self.morph_transformation(color_segmentation_mask)
		#cv2.imshow('after morph',pixel_candidates)

		pixel_candidates        = self.ccl_generation_filtering(pixel_candidates)
		#cv2.imshow('after ccl',pixel_candidates)
		#cv2.waitKey()
		return pixel_candidates