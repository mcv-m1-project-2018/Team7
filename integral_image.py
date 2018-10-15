import cv2
import numpy as np


class IntegralImage():
    def __init__(self, image):
        self.integral_image = cv2.integral(image)[1:,1:]


    def get_integral_image(self):
        return self.integral_image

    # vertices of the rectangle are ordered as follows: (* pixels are the ones to sum)
    #    A----------B
    #    |***********
    #    |***********
    #    C***********<-D
    #this function returns the sum of the pixel values within the rectangle
    def calculate_value_rectangle(self, A, B, C, D):
        return self.integral_image[D] - self.integral_image[B] - self.integral_image[C] + self.integral_image[A]



######### Example ############################################################
snap = np.ones((3,3),dtype='uint8')
print(snap)

integral_image = IntegralImage(snap)
print(integral_image.get_integral_image())
print("---------")
print(integral_image.calculate_value_rectangle((0, 0), (0, 2), (2, 0), (2, 2)))
