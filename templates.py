
#================================================================================================
# data.py

class instance(img, msk, bboxes, sign_type, img_id):
	img       = None       # numpy array  ==> rgb
	msk       = None       # numpy array  ==> binary image 0 or 1
	bboxes    = None       #[ [tly, tlx, bry, brx], ... ]
	sign_type = None       #[         'F'         , ... ]
	img_id    = None       # srting


class data_handler():
	def __init__(self, train_dir = './train/', test_dir = './test/', results = './results/'):
		self.train_set = []   # [instance(), ...]
		self.valid_set = []   # [instance(), ...]
		self.test_set  = []   # [instance(), ...]
		self.types     = []   # ['A','B','C','D','E','F']

	def read_all():
		return self.train_set, self.valid_set, self.test_set

	def parse_sign_type(sign_type):
		#parse_sign_type(sign_type)   in db_analysis.py
		return description
#================================================================================================
# data_analysis.py



class data_analysis():

	def shape_analysis(train_set):
		"""
		sign_count, max_area, min_area, filling_ratios: all are dic and keys are data_handler.types		
		"""
		return sign_count, max_area, min_area, filling_ratios   

	def color_analysis(train_set):
		"""
		return mean and std of distributions and any other useful info for detection and recognition
		"""
		pass



class detector()
	def 







"""
args = docopt(__doc__)

images_dir = args['<dirName>']          # Directory with input images and annotations
output_dir = args['<outPath>']          # Directory where to store output masks, etc. For instance '~/m1-results/week1/test'
pixel_method = args['<pixelMethod>']
window_method = args['--windowMethod']

pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity, window_precision, window_accuracy = 
traffic_sign_detection(images_dir, output_dir, pixel_method, window_method);

print(pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity, window_precision, window_accuracy)
"""