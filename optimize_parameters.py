import numpy as np
import cv2
import imageio
import time
from data import Data_handler
from traffic_signs.evaluation.evaluation_funcs import performance_accumulation_pixel, performance_accumulation_window
from traffic_signs.evaluation.evaluation_funcs import performance_evaluation_pixel, performance_evaluation_window


def morph_transformation(pixel_candidates):
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

        x, y, w, h = cv2.boundingRect(contour)

        width = y_max - y_min
        height = x_max - x_min

        # check if the aspect ratio and area are bigger or smaller than the ground truth. If it is consistent with
        # the ground truth, we try to fill it (some signs are not fully segmented because they contain white or other
        # colors) with cv2.fillPoly.
        if max_aspect_ratio > height / width > min_aspect_ratio and max_area > width * height > min_area:
            cv2.fillPoly(pixel_candidates, pts=[contour], color=255)
            #cv2.rectangle(pixel_candidates, (x, y), (x + w, y + h), 180, 2)
        else:
            cv2.fillPoly(pixel_candidates, pts=[contour], color=0)
            #cv2.rectangle(pixel_candidates, (x, y), (x + w, y + h), 50, 2)


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

def pixel_method(im,
                 blue_low_hsv=(105, 30, 30),
                 blue_high_hsv=(135, 255, 255),
                 red1_low_hsv=(0, 50, 50),
                 red1_high_hsv=(8, 255, 255),
                 red2_low_hsv=(177, 50, 50),
                 red2_high_hsv=(181, 255, 255)
                 ):

    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    blue_low = np.array(blue_low_hsv, dtype='uint8')
    blue_high = np.array(blue_high_hsv, dtype='uint8')

    red_1_low = np.array(red1_low_hsv, dtype='uint8')
    red_1_high = np.array(red1_high_hsv, dtype='uint8')

    red_2_low = np.array(red2_low_hsv, dtype='uint8')
    red_2_high = np.array(red2_high_hsv, dtype='uint8')

    mask_hue_red_1 = cv2.inRange(hsv_image, red_1_low, red_1_high)
    mask_hue_red_2 = cv2.inRange(hsv_image, red_2_low, red_2_high)

    combined_red_mask = cv2.bitwise_or(mask_hue_red_1, mask_hue_red_2)
    mask_hue_blue = cv2.inRange(hsv_image, blue_low, blue_high)
    final_mask = cv2.bitwise_or(combined_red_mask, mask_hue_blue)

    return final_mask // 254

def evaluate_parameters(train_split, blue_low_hsv, blue_high_hsv, red1_low_hsv, red1_high_hsv, red2_low_hsv, red2_high_hsv):

    pixelTP, pixelTN, pixelFP, pixelFN = 0, 0, 0, 0
    for image_instance in train_split:
        image = cv2.imread(image_instance.img)
        color_segmentation_mask = pixel_method(image,
                        blue_low_hsv=blue_low_hsv,
                        blue_high_hsv=blue_high_hsv,
                        red1_low_hsv=red1_low_hsv,
                        red1_high_hsv=red1_high_hsv,
                        red2_low_hsv=red2_low_hsv,
                        red2_high_hsv=red2_high_hsv)

        pixel_candidates = morph_transformation(color_segmentation_mask)

        pixel_annotation = imageio.imread('{}/mask/mask.{}.png'.format(directory, image_instance.img_id)) > 0

        [localPixelTP, localPixelFP, localPixelFN, localPixelTN] = performance_accumulation_pixel(
            pixel_candidates=pixel_candidates,
            pixel_annotation=pixel_annotation)
        pixelTP = pixelTP + localPixelTP
        pixelFP = pixelFP + localPixelFP
        pixelFN = pixelFN + localPixelFN
        pixelTN = pixelTN + localPixelTN

    return performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN)


def save_progress(parameters, parameter, t1, current_max_value, current_precision, current_sensitivity):
    with open('optimization_parameters.log', "a") as f:
        f.write("-------------------------------- "+parameter+"\n")
        f.write("total time: "+str((time.time()-t1)/60)+" min\n")
        f.write(str(parameters)+"\n")
        f.write("Score: " +str(current_max_value)+"\n")
        f.write("Precision: " + str(current_precision)+"\n")
        f.write("Sensitivity: " + str(current_sensitivity)+"\n")
    print("-------------------------------- " + parameter)
    print("total time: " + str((time.time()-t1)/60)+" min\n")
    print(str(parameters))
    print("Score: " + str(current_max_value))
    print("Precision: " + str(current_precision))
    print("Sensitivity: " + str(current_sensitivity))


############################################## OPTIMIZATION HYPER-PARAMETERS ################################
def score(precision, sensitivity):
    return 0.7*sensitivity+0.3*precision

MAX_RANGE = 40

##############################################################################################################

#### DATA LOADING ############################################################################################
data_handler = Data_handler()
train_set, valid_set, test_set = data_handler.read_all()

train_split = train_set
directory = "./train/"

parameters = { #[optimal_value, start_range, end_range]
    'blue_low_h': [105, 90, 140],
    'blue_low_s': [30, 20, 255],
    'blue_low_v': [30, 20, 255],
    'blue_high_h': [135, 90, 140],
    'blue_high_s': [255, 20, 255],
    'blue_high_v': [255, 20, 255],
    'red1_low_h': [0, 0, 25],
    'red1_low_s': [50, 20, 255],
    'red1_low_v': [50, 20, 255],
    'red1_high_h': [8, 0, 25],
    'red1_high_s': [255, 20, 255],
    'red1_high_v': [255, 20, 255],
    'red2_low_h': [177, 165, 180],
    'red2_low_s': [50, 20, 255],
    'red2_low_v': [50, 20, 255],
    'red2_high_h': [180, 165, 180],
    'red2_high_s': [255, 20, 255],
    'red2_high_v': [255, 20, 255],
}

t1 = time.time()

blue_low_hsv = (parameters['blue_low_h'][0], parameters['blue_low_s'][0], parameters['blue_low_v'][0])
blue_high_hsv = (parameters['blue_high_h'][0], parameters['blue_high_s'][0], parameters['blue_high_v'][0])
red1_low_hsv = (parameters['red1_low_h'][0], parameters['red1_low_s'][0], parameters['red1_low_v'][0])
red1_high_hsv = (parameters['red1_high_h'][0], parameters['red1_high_s'][0], parameters['red1_high_v'][0])
red2_low_hsv = (parameters['red2_low_h'][0], parameters['red2_low_s'][0], parameters['red2_low_v'][0])
red2_high_hsv = (parameters['red2_high_h'][0], parameters['red2_high_s'][0], parameters['red2_high_v'][0])

[pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity] = evaluate_parameters(
    train_split=train_split,
    blue_low_hsv = blue_low_hsv,
    blue_high_hsv = blue_high_hsv,
    red1_low_hsv = red1_low_hsv,
    red1_high_hsv = red1_high_hsv,
    red2_low_hsv = red2_low_hsv,
    red2_high_hsv = red2_high_hsv
    )

current_max_value = score(precision=pixel_precision, sensitivity=pixel_sensitivity)

########################################## OPTIMIZATION ###############################################################

for parameter in parameters:

    print("*************************************")
    print("optimizing  "+parameter+"  ...")
    current_parameter = parameters[parameter][0]
    start_range = max(parameters[parameter][1], current_parameter - MAX_RANGE//2)
    end_range = min(parameters[parameter][2], current_parameter + MAX_RANGE//2)

    for p in range(start_range, end_range):
        if(p!=current_parameter):
            print(p)
            parameters[parameter][0] = p
            blue_low_hsv = (parameters['blue_low_h'][0], parameters['blue_low_s'][0], parameters['blue_low_v'][0])
            blue_high_hsv = (parameters['blue_high_h'][0], parameters['blue_high_s'][0], parameters['blue_high_v'][0])
            red1_low_hsv = (parameters['red1_low_h'][0], parameters['red1_low_s'][0], parameters['red1_low_v'][0])
            red1_high_hsv = (parameters['red1_high_h'][0], parameters['red1_high_s'][0], parameters['red1_high_v'][0])
            red2_low_hsv = (parameters['red2_low_h'][0], parameters['red2_low_s'][0], parameters['red2_low_v'][0])
            red2_high_hsv = (parameters['red2_high_h'][0], parameters['red2_high_s'][0], parameters['red2_high_v'][0])

            [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity] = evaluate_parameters(
                train_split=train_split,
                blue_low_hsv=blue_low_hsv,
                blue_high_hsv=blue_high_hsv,
                red1_low_hsv=red1_low_hsv,
                red1_high_hsv=red1_high_hsv,
                red2_low_hsv=red2_low_hsv,
                red2_high_hsv=red2_high_hsv
                )
            value = score(precision=pixel_precision, sensitivity=pixel_sensitivity)
            print(value)
            if (value > current_max_value):
                current_parameter = p
                current_max_value = value
                current_precision = pixel_precision
                current_sensitivity = pixel_sensitivity

    parameters[parameter][0] = current_parameter
    save_progress(parameters=parameters,
                  parameter=parameter,
                  t1=t1,
                  current_max_value=current_max_value,
                  current_precision=current_precision,
                  current_sensitivity=current_sensitivity)





