import cv2
import numpy as np
import os
from Data_augmentation import Data_augmentation


def load_image(path,input_size):
    # load image function
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # remove noise
    img = cv2.GaussianBlur(gray,(3,3),0)

    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()

    # we crop image from center
    short_edge = min(img.shape[:2]) 
    yy = int((img.shape[0] - short_edge) / 2)	    
    xx = int((img.shape[1] - short_edge) / 2)
    img = img[yy: yy + short_edge, xx: xx + short_edge]

    #img = Data_augmentation(cv2.resize(img, (input_size[0],input_size[1]), interpolation = cv2.INTER_AREA))
				

	# resize to 32, 32
    return cv2.resize(img, (input_size[0],input_size[1]), interpolation = cv2.INTER_AREA)

def save_test_data(x,y,directory):
    ##Save test data images

    os.chdir('/Users/Theologis/Desktop/CAS Lab/test_data/') # to specified directory 
    print("Saving Test Data")
    for i in range(len(x)):
        img_name = str(i)+ str(y[i]) +'.png'
        cv2.imwrite(img_name, x[i]* 255.0)

    os.chdir(directory) # to specified directory  


def myAtoi(string):
    #https://www.geeksforgeeks.org/write-your-own-atoi/
    res = 0
 
    # Iterate through all characters of
    #  input string and update result
    for i in range(len(string)):
        res = res * 10 + (ord(string[i]) - ord('0'))
 
    return res

def scale(x, feature_range=(-1, 1)):
    ## not used
    # scale to (0, 1)
    x = ((x - x.min())/(255 - x.min()))
    
    # scale to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x
