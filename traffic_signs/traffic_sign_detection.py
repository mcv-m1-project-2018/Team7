#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Usage:
  traffic_sign_detection.py <dirName> <outPath> <pixelMethod> [--windowMethod=<wm>] 
  traffic_sign_detection.py -h | --help
Options:
  --windowMethod=<wm>        Window method       [default: None]
"""


import fnmatch
import os
import sys
import pickle
import random

import numpy as np
import imageio
from docopt import docopt
import cv2

from .candidate_generation_pixel  import candidate_generation_pixel
from .candidate_generation_window import candidate_generation_window
from .evaluation.load_annotations import load_annotations
from .evaluation.evaluation_funcs import performance_accumulation_pixel, performance_accumulation_window
from .evaluation.evaluation_funcs import performance_evaluation_pixel, performance_evaluation_window


def traffic_sign_detection(split, directory, ids, output_dir, pixel_method, window_method, show_progress=False):
    # -1 just to avoid division by zero
    pixelTP = 1
    pixelFN = 1
    pixelFP = 1
    pixelTN = 1

    windowTP = 0
    windowFN = 0
    windowFP = 0

    window_precision = 0
    window_accuracy  = 0

    # Load image names in the given directory
    random.shuffle(ids)

    if split == "val":
        for id_ in ids:

            # Read file
            image = cv2.imread('{}/{}'.format(directory, id_ + ".jpg"))

            if show_progress:
                print('{}/{}'.format(directory,id_+".jpg"))

            # Candidate Generation (pixel) ######################################
            pixel_candidates = candidate_generation_pixel(image, pixel_method)

            fd = '{}/{}_{}_{}'.format(output_dir, pixel_method, window_method, split)
            if not os.path.exists(fd):
                os.makedirs(fd)

            out_mask_name = '{}/{}.png'.format(fd, id_)
            imageio.imwrite (out_mask_name, np.uint8(np.round(pixel_candidates)))


            if window_method != 'None':
                window_candidates = candidate_generation_window(image, pixel_candidates, window_method)

                out_list_name = '{}/{}.pkl'.format(fd, id_)

                with open(out_list_name, "wb") as fp:   #Pickling
                    pickle.dump(window_candidates, fp)



            # Accumulate pixel performance of the current image #################
            pixel_annotation = imageio.imread('{}/mask/mask.{}.png'.format(directory,id_)) > 0

            [localPixelTP, localPixelFP, localPixelFN, localPixelTN] = performance_accumulation_pixel(pixel_candidates, pixel_annotation)
            pixelTP = pixelTP + localPixelTP
            pixelFP = pixelFP + localPixelFP
            pixelFN = pixelFN + localPixelFN
            pixelTN = pixelTN + localPixelTN

            [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity] = performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN)

            if window_method != 'None':
                # Accumulate object performance of the current image ################
                window_annotationss = load_annotations('{}/gt/gt.{}.txt'.format(directory, id_))
                [localWindowTP, localWindowFN, localWindowFP] = performance_accumulation_window(window_candidates, window_annotationss)

                windowTP = windowTP + localWindowTP
                windowFN = windowFN + localWindowFN
                windowFP = windowFP + localWindowFP


                # Plot performance evaluation
                [window_precision, window_sensitivity, window_accuracy] = performance_evaluation_window(windowTP, windowFN, windowFP)
    
        return [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity, window_precision, window_accuracy]
    if split == "test":
        for id_ in ids:

            # Read file
            image = cv2.imread('{}/{}'.format(directory, id_ + ".jpg"))

            if show_progress:
                print ('{}/{}'.format(directory,id_+".jpg"))

            # Candidate Generation (pixel) ######################################
            pixel_candidates = candidate_generation_pixel(image, pixel_method)

            fd = '{}/{}_{}_{}'.format(output_dir, pixel_method, window_method, split)
            if not os.path.exists(fd):
                os.makedirs(fd)

            out_mask_name = '{}/{}.png'.format(fd, id_)
            imageio.imwrite (out_mask_name, np.uint8(np.round(pixel_candidates)))
    return


if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)

    images_dir = args['<dirName>']          # Directory with input images and annotations
    output_dir = args['<outPath>']          # Directory where to store output masks, etc. For instance '~/m1-results/week1/test'
    pixel_method = args['<pixelMethod>']
    window_method = args['--windowMethod']

    pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity, window_precision, window_accuracy = traffic_sign_detection(images_dir, output_dir, pixel_method, window_method);

    print(pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity, window_precision, window_accuracy)

    
