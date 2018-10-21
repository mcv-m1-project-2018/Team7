import os

import cv2
import numpy as np

from traffic_sign_model import Traffic_sign_model


class TemplateMatching(Traffic_sign_model):
    def window_method(self, im, pixel_candidates):
        # Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
        final_mask, window_candidates = self.template_matching(im, pixel_candidates, threshold=.50, show=False)

        import matplotlib.pyplot as plt
        import matplotlib.patches as pat
        plt.imshow(im)
        for window in window_candidates:
            rec = pat.Rectangle((window[0], window[1]), window[2], window[3])
            







        return window_candidates

    def template_matching(self, im, pixel_candidates, threshold=.45, show=False):
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
            padding = 10
            # cops the interest region + a padding
            region = im[max(0, y_min - padding):min(im.shape[0], y_max + padding),
                        max(0, x_min - padding):min(im.shape[1], x_max + padding)]
            region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY).astype(np.float32)

            mask_region = pixel_candidates[max(0, y_min - padding):min(im.shape[0], y_max + padding),
                                           max(0, x_min - padding):min(im.shape[1], x_max + padding)]

            region_masked = cv2.bitwise_and(region, region, mask=mask_region)

            dsize = min(region.shape)
            max_score = 0
            scalars = [1]  # this is to give different scales to the template, not sure if we should use it

            # print((dsize, dsize), ", ", region.shape)

            for template in templates:
                for scalar in scalars:
                    dsize_scaled = int(dsize * scalar)
                    template_g = cv2.resize(template, dsize=(dsize_scaled, dsize_scaled), interpolation=cv2.INTER_CUBIC)
                    res = cv2.matchTemplate(region_masked, template_g, cv2.TM_CCOEFF_NORMED)
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

        window_candidates = self.get_ccl_bbox(pixel_candidates)

        return final_mask, window_candidates


    def ccl_generation_filtering(self, pixel_candidates):
        # find all contours (segmented areas) of the mask to delete those that are not consistent with the train split
        # analysis

        max_aspect_ratio = 1.419828704905269 * 1.25
        min_aspect_ratio = 0.5513618362563639 * 0.75
        max_area = 55919.045 * 1.15
        min_area = 909.7550000000047 * 0.75
        max_filling_ratio = 0.40

        image, contours, hierarchy = cv2.findContours(pixel_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            # max and min coordinates of the segmented area
            x, y, width, height = cv2.boundingRect(contour)

            # check if the aspect ratio and area are bigger or smaller than the ground truth. If it is consistent with
            # the ground truth, we try to fill it (some signs are not fully segmented because they contain white or
            # other colors) with cv2.fillPoly.
            if max_aspect_ratio > width/height > min_aspect_ratio and max_area > width*height > min_area:
                cv2.fillPoly(pixel_candidates, pts=[contour], color=255)
            else:
                cv2.fillPoly(pixel_candidates, pts=[contour], color=0)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        pixel_candidates = cv2.dilate(pixel_candidates, kernel, iterations=1)

        image, contours, hierarchy = cv2.findContours(pixel_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)

            pixel_count = 0
            for x_ in range(x, x+width):
                for y_ in range(y, y+height):
                    if pixel_candidates[y_, x_]:
                        pixel_count += 1
            filling_ratio = pixel_count / (width * height)

            print(filling_ratio)

            if max_aspect_ratio > width/height > min_aspect_ratio and max_area > width*height > min_area and \
                    filling_ratio > 0.4:
                cv2.fillPoly(pixel_candidates, pts=[contour], color=255)
            else:
                cv2.fillPoly(pixel_candidates, pts=[contour], color=0)

        return pixel_candidates



    def __init__(self):
        Traffic_sign_model.__init__(self)
        self.window_method_name = '3_templatematching'
        self.pixel_method_name = 'hsvmorph'

        self.parameters = {
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


model = TemplateMatching()
model.evaluate(split='val', output_dir='test_results/')





















