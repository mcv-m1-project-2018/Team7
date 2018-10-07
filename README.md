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



