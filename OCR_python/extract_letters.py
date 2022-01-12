import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random as rng
from extract_utils import doOverlap,in_Same_line
from utils import BoundingBoxes_from_Points

def FindLetters(start_point,end_point,img_dir):
    #Find letters function in 4 steps that finds each characher and cuts the noise.

    #Reading image
    image = cv2.imread(img_dir)

    start_point = np.array(start_point)
    end_point = np.array(end_point)
    #Keep near boxes
    start_point,end_point = Keep_near_boxes(start_point,end_point)

    #Delete to big boxes
    start_point,end_point = delete_big_boxes(start_point,end_point,image)

    boxes = np.zeros((start_point[:,0].size, 4), dtype="float32")
    boxes[:,0] = start_point[:,0]
    boxes[:,1] = start_point[:,1]
    boxes[:,2] = end_point[:,0]
    boxes[:,3] = end_point[:,1]

    #Non-mas supression
    boxes = non_max_suppression_slow(boxes, .5)

    start_point = np.zeros((boxes[:,0].size,2), dtype="int32")
    start_point[:,0] = boxes[:,0]
    start_point[:,1] = boxes[:,1]
    end_point = np.zeros((boxes[:,0].size,2), dtype="int32")
    end_point[:,0] = boxes[:,2]
    end_point[:,1] = boxes[:,3]

    BoundingBoxes_from_Points(start_point,end_point,image)
    print("Detected boxes:",len(start_point))

    return start_point, end_point

def delete_big_boxes(start_point,end_point,image):
    ## 1) Get rid of the 'inside counts' function
    start_point_np = np.array(start_point)
    end_point_np = np.array(end_point)
    delete_point = [] # j points that was deleted 
    for i in range(len(start_point)): 
        if (( (end_point[i][0]- start_point[i][0])> image.shape[1]/3  or (end_point[i][1]- start_point[i][1])> image.shape[0]/3 )
             ) :
            delete_point.append(i) #save the deleted i

    delete_point.sort()
    count_deleted = 0 # how many points have been deleted 
    for i in delete_point:
        #delete the points
        start_point_np = np.delete(start_point_np, i-count_deleted, 0)
        end_point_np = np.delete(end_point_np, i-count_deleted, 0)
        count_deleted +=1 

    return start_point_np,end_point_np

def non_max_suppression_slow(boxes, overlapThresh):


	# initialize the list of picked indexes
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]

		# loop over all indexes in the indexes list
		for pos in range(0, last):
			# grab the current index
			j = idxs[pos]

			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])

			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)

			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / area[j]

			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > overlapThresh:
				suppress.append(pos)

		# delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)

	# return only the bounding boxes that were picked
	return boxes[pick]


def Keep_near_boxes(start_point,end_point):
    #keep those that have other boxes near by and close Υ point values(two letters in the same line)
    start_point_np = np.array(start_point)
    end_point_np = np.array(end_point)
    potition=0
    for i in range(len(start_point)):  
        for j in range(len(start_point)):
            if (i!=j and in_Same_line(start_point[i],end_point[i],start_point[j],end_point[j])): #if they are in the same line 
                distance = min(np.linalg.norm(end_point[i]-(start_point[j][0],end_point[j][1])),np.linalg.norm(end_point[j]-(start_point[i][0],end_point[i][1]))) #their distance
                min_distance =max(end_point[j][1]-start_point[j][1],end_point[i][1] -start_point[i][1]) #adaptive theshold distance in  px
                if (distance <= min_distance): #if they are close save i box
                    start_point_np[potition] = start_point[i]
                    end_point_np[potition] = end_point[i]
                    potition +=1
                    break

    return start_point_np[:potition],end_point_np[:potition]
