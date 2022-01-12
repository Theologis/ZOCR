import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random as rng
from Data_augmentation import Data_augmentation


image_folder = "images/"

def RGBtoHSV(proj_directory,image_name,sensitivity):
    ##Step 1 
    path = proj_directory + image_folder + image_name
    RGBimag = cv2.imread(path) 
    hsv = cv2.cvtColor(RGBimag, cv2.COLOR_BGR2HSV)
    
    
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([179,sensitivity,255])

    
    
    mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(RGBimag,RGBimag, mask= mask)


    plt.figure(figsize=(18,18))
    plt.imshow(res)
    plt.axis('off')
    plt.show()
    return res


def load_image(path,input_size,data_augm=False):
    # load image function
    img = cv2.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    #data augmentation
    if data_augm :
        # we crop image from center
        short_edge = min(img.shape[:2]) 
        yy = int((img.shape[0] - short_edge) / 2)	    
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
        img = crop_img
        img = Data_augmentation(cv2.resize(crop_img, (input_size[0],input_size[1]), interpolation = cv2.INTER_AREA))
				

	# resize to 32, 32
    return cv2.resize(img, (input_size[0],input_size[1]), interpolation = cv2.INTER_AREA)

def BoundingBoxes_from_contours(contours,canny_output):
    ##Creating Bounding boxes  from contours
    ###Source : https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
            
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    start_point = []
    end_point = []
    # Draw polygonal contour + bonding rects 
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing, contours_poly, i, color)
        cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
            (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
            
        start_point.append((int(boundRect[i][0]), int(boundRect[i][1]))) #start point of the rectangle
        end_point.append((int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3]))) ##end point of the rectangle
            
    #create an empty image for contours
    img_contours = np.zeros(canny_output.shape)
    # draw the contours on the empty image
    """
    cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)
    plt.figure(figsize=(18,18))
    plt.imshow(drawing)
    plt.axis('off')
    plt.show()
    """
    return start_point,end_point


def BoundingBoxes_from_Points(start_point,end_point,canny_output):
    #Creating  and draw Bounding boxes  from start and end point function
    for i in range(len(start_point)):
        #color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        color = (102,2, 60)
        cv2.rectangle(canny_output,(start_point[i][0], start_point[i][1]) , \
                                   (end_point[i][0], end_point[i][1]),
                                   color,2)
    # draw the contours on the empty image
    plt.figure(figsize=(18,18))
    plt.imshow(canny_output)
    plt.axis('off')
    plt.show()

def BoundingBoxes_from_Points_with_labels(start_point,end_point,canny_output,predections,P_letter_is_real_letter):
    #Creating  and draw Bounding boxes  from start and end point function
    for i in range(len(start_point)):
        if P_letter_is_real_letter[i] > .99:
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv2.rectangle(canny_output,(start_point[i][0], start_point[i][1]) , \
                                    (end_point[i][0], end_point[i][1]),
                                    color,2)
            cv2.putText(canny_output,str(predections[i]) + ":" + " " + str(np.round(P_letter_is_real_letter[i],2)*100) + "%", (start_point[i][0], start_point[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4,color = color, thickness = 1)
    # draw the contours on the empty image
    plt.figure(figsize=(18,18))
    plt.imshow(canny_output)
    plt.axis('off')
    plt.show()

def GetBoundingBoxes(img,threshold_area):
    
    #convert img to grey step2
    img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #start of step 3 :
    #set a thresh
    thresh = 100
    #get threshold image
    ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    #find contours
    _,contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    thres_cnt = []
    #get rid of the small contours
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > threshold_area:
            thres_cnt.append(cnt)
    #draw the contours
    start_point,end_point = BoundingBoxes_from_contours(thres_cnt,img)
    return start_point,end_point

def MSER(img_dir):
    path = img_dir

    mser = cv2.MSER_create()

    #Reading image
    img = cv2.imread(path)

    #Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    vis = img.copy()

    #Detect regions in gray scale image
    regions, _ = mser.detectRegions(gray)

    #Hulls for each dtected regions
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    #Drawing polylines on image
    cv2.polylines(img, hulls, 1, (0, 255, 0))


    start_point,end_point = BoundingBoxes_from_contours(hulls,img)
    BoundingBoxes_from_Points(start_point,end_point,vis)

    print("Detected boxes:",np.array(end_point, dtype=np.float32).size)

    return start_point,end_point

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
