# Team7 - Project 1: Traffic Sign Detection/Recognition

This is our code for the first project. 

## Requirements
 
The file ```requirements.txt``` contains all the required packages to run our code. This project has been developed in Python
3.7.

To use the notebook you need to place the ```./train/``` and the ```./test/``` folders in the root folder of this project. 


## Running the code
To run our code simply open the notebook ``` workflow.ipynb ``` and run each part to read, analyze or test on data.

Alternatively you can run the ```traffic_sign_model.py``` file and specify the location of the data folders. Use ```python 
traffic_sign_model.py -h``` to see the optional arguments.  

## Results
The masks of the segmented images are stored in ```OUTPUT_DIR/PIXELMETHOD_WINDOWMETHOD_SPLIT``` where ```PIXELMETHOD``` 
and  ```WINDOWMETHOD``` are the selected pixel and window methods, and ```SPLIT``` is either the validation or the
test split. 

### Random methods
I use this to keep track of some methods and the values used in them, just so I don't forget them. The names of the 
methods are random, ignore them. <br />
Method1, Method2, Method3, use different values in the kernel used in the last dilatation. 

#### Method1

* Pixel Segmentation Values:

```
self.blue_low_hsv = (105, 30, 30)
self.blue_high_hsv = (135, 255, 255)
self.red1_low_hsv = (0, 50, 50)
self.red1_high_hsv = (8, 255, 255)
self.red2_low_hsv = (177, 50, 50)
self.red2_high_hsv = (181, 255, 255)
```

* Morphological operations:

```
Open - kernel 5, 5, cv2.MORPH_ELLIPSE
Close - kernel 10, 10, cv2.MORPH_ELLIPSE
CCL discarding + cv2.fillPoly
Dilate - kernel 10, 10, cv2.MORPH_ELLIPSE
CCL discarding + cv2.fillPoly
```

* No window method <br />

Scores: 

Precision: 0.50 <br />
Sensitivity: 0.81 <br />
Window Precision: 0.5  <br />
Window Accuracy: 0.24 <br />

#### Method2

* Pixel Segmentation Values:

```
self.blue_low_hsv = (105, 30, 30)
self.blue_high_hsv = (135, 255, 255)
self.red1_low_hsv = (0, 50, 50)
self.red1_high_hsv = (8, 255, 255)
self.red2_low_hsv = (177, 50, 50)
self.red2_high_hsv = (181, 255, 255)
```

* Morphological operations:

```
Open - kernel 5, 5, cv2.MORPH_ELLIPSE
Close - kernel 10, 10, cv2.MORPH_ELLIPSE
CCL discarding + cv2.fillPoly
Dilate - kernel 20, 20, cv2.MORPH_ELLIPSE
CCL discarding + cv2.fillPoly
```

* No window method <br />

Scores: 

Precision: 0.41 <br />
Sensitivity: 0.84 <br />
Window Precision: 0.5  <br />
Window Accuracy: 0.24 <br />


#### Method3

* Pixel Segmentation Values:

```
self.blue_low_hsv = (105, 30, 30)
self.blue_high_hsv = (135, 255, 255)
self.red1_low_hsv = (0, 50, 50)
self.red1_high_hsv = (8, 255, 255)
self.red2_low_hsv = (177, 50, 50)
self.red2_high_hsv = (181, 255, 255)
```

* Morphological operations:

```
Open - kernel 5, 5, cv2.MORPH_ELLIPSE
Close - kernel 10, 10, cv2.MORPH_ELLIPSE
CCL discarding + cv2.fillPoly
Dilate - kernel 7, 7, cv2.MORPH_ELLIPSE
CCL discarding + cv2.fillPoly
```

* No window method <br />

Scores: 

Precision: 0.54 <br />
Sensitivity: 0.78 <br />
Window Precision: 0.5  <br />
Window Accuracy: 0.24 <br />












