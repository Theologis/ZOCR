import cv2
import os
import numpy as np

def doOverlap(l1, r1, l2, r2):
    #https://www.geeksforgeeks.org/find-two-rectangles-overlap/
    # Returns true if two rectangles(l1, r1)  
    # and (l2, r2) overlap

    x=0
    y=1

    if (l1 == l2).all() and (r1 == r2).all():
        return True

    # If one rectangle is on left up side of other 
    if((l1<=r2).all() and (r1>=r2).all()) or ((l2<=r1).all() and (r2>=r1).all()): 
        return True
    # If one rectangle is on right down side of other 
    if ((l1>=l2).all() and (l1<=r2).all()) or ((l2>=l1).all() and (l2<=r1).all()) : 
        return True
    # If one rectangle is on right up side of other
    if((r1[x]<=l2[x] and r2[y]<=l1[y]) and (r2[x]<=r1[x] and l1[y]<=l2[y])) or ((r2[x]<=l1[x] and r1[y]<=l2[y]) and (r1[x]<=r2[x] and l2[y]<=l1[y])) :
        return True
    # If one rectangle is on left down side of other
    if((l1[x]>=l2[x] and r1[y]<=r2[y]) and (l2[x]>=r1[x] and r2[y]<=l1[y])) or ((l2[x]>=l1[x] and r2[y]<=r1[y]) and (l1[x]>=r2[x] and r1[y]<=l2[y]))  :
        return True
  
    return False

def in_Same_line(start_pointA,end_pointA,start_pointB,end_pointB):
    ###find if Box A and B have close Υ point values .

    up_threshold = max(end_pointA[1]-start_pointA[1],end_pointB[1]-start_pointB[1])/2  #Calculare the adaprive thres.....
    down_threshold = max(end_pointA[1]-start_pointA[1],end_pointB[1]-start_pointB[1])/5

    if ((start_pointA[1]<=start_pointB[1] + up_threshold and start_pointA[1]>=start_pointB[1]-up_threshold ) and
        (end_pointA[1]<=end_pointB[1]+down_threshold and end_pointA[1]>=end_pointB[1]-down_threshold) ):
        return True

    return False
