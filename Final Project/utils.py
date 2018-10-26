#coding=utf-8
import cv2
import matplotlib.pyplot as plt
import tarfile
import os
import numpy as np
import cv2 as cv
from skimage import morphology
from scipy.ndimage.morphology import binary_fill_holes
from scipy.linalg import norm
import math
import scipy
import glob
from scipy import linalg
import imutils
import skimage.io
from operator import itemgetter
import matplotlib.pyplot as plt

def find_circle(hole_contours):
    ''' 
    Find the circle's contour by searching for the more compact shape.
    
    << INPUT >>
        hole_contours: Contour list of holes
        
    << OUTPUT>>  
        hole_contours: List without the circle
        circle_center: Location of the circle(Center)
    '''
    compacity = []

    # Compute the compacity of each holes
    for c in hole_contours:
        area = cv.contourArea(c)
        perimeter = cv.arcLength(c,True)
        compacity.append(perimeter**2/area)
    compacity = np.asarray(compacity)
    print('The computed compacities of inputs are ', compacity)

    # Find the minimum compacity among holes
    circle_idx = np.argmin(compacity)
    print('The index of the circle is ', circle_idx)
    
    # Remove the circle from the input list
    circle = hole_contours.pop(circle_idx)
    
    # Find the location of the circle
    circle_center = sum(circle)/len(circle)
    
    return [hole_contours, circle_center]


def detect_shape(c):
    '''
    INPUT : Contour of the object.
    
    OUTPUT: Shape of the object in a string
            Area(size) of the object
    '''
    # initialize the shape name and approximate the contour
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.04 * peri, True)
    area = cv.contourArea(c)
    center = []
    
    # If there are 3 vertices, the shape is Triangle
    if len(approx) == 3:
        shape = "triangle"
        M = cv.moments(c)
        c_width = int((M["m10"] / M["m00"]))
        c_height = int((M["m01"] / M["m00"]))
        center = (c_width,c_height)
    else:
        shape = "Others"
        
    # Return the name of the shape an its size
    return shape, center


def sort_dict_by_first_value_tuple(dictionary):
    '''
      Sort unordered dictionary of locations of pairs of shapes based on the corresponding digit
    '''
    return sorted(dictionary.items(), key = itemgetter(1))


def make_dict_from_list(dictionary):
    '''
    Given location data as a dictionary type, returns two list of ordered dictionary
    
    ordered_digit : a list of digits on shapes and corresponding holes
    ordered_locations: a list of ordered locations that we need to visit
    
    '''
    ordered = sort_dict_by_first_value_tuple(dictionary)
    
    # Sort the detected numbers in an ascending order 
    ordered_digit = sorted([val[1][0] for x,val in enumerate(ordered)]*2)
    
    # get locations of shape and wholes in common list
    ordered_loc = list(val[1][1:] for x,val in enumerate(ordered))
    ordered_point_1loc = [val[0] for val in (ordered_loc)] # locations of shapes
    ordered_point_2loc = [val[1] for val in (ordered_loc)] # locations of the holes
    
    # Put the locations of matched pieces and holes that correspond to the numbers
    ordered_locations = [None]*(len(ordered_digit))
    for i in range(int(len(ordered_digit)/2)): 
        ordered_locations[2*i] = ordered_point_1loc[i]
        ordered_locations[2*i+1] = ordered_point_2loc[i]
        
    return ordered_digit, ordered_locations

def growingRegion(image, start, margin):
    '''
        Given image and starting points in the image, 
        this function uses region growing method to detect an object.     
    '''
    
    # Initialize the variables
    visited_pixels = np.zeros(image.shape);
    region_pixels = [];
    tovisit_pixels = [];
    reference_values = [];
    
    # Get the image information
    w =image.shape[1]
    h =image.shape[0]
    
    tovisit_pixels.append(start);
    reference_values.append(image.item(start[1],start[0]));
    
    # Do region growing
    while(len(tovisit_pixels)>0):
        pixel = tovisit_pixels.pop(0);
        ref_value = reference_values.pop(0);
        if (visited_pixels.item(pixel[1],pixel[0])==0):
            value = image.item(pixel[1],pixel[0]);
            visited_pixels.itemset((pixel[1],pixel[0]),1);
            
            if(abs(value-ref_value)<margin):
                region_pixels.append(pixel);
                if(pixel[0]+1<w):
                    tovisit_pixels.append((pixel[0]+1,pixel[1]));
                    reference_values.append(value);
                if(pixel[1]+1<h):
                    tovisit_pixels.append((pixel[0],pixel[1]+1));
                    reference_values.append(value);
                if(pixel[1]-1>0):
                    tovisit_pixels.append((pixel[0],pixel[1]-1));
                    reference_values.append(value); 
                if(pixel[0]-1>0):
                    tovisit_pixels.append((pixel[0]-1,pixel[1]));
                    reference_values.append(value);    
                    
    return region_pixels


    
def rad2degree(angle_rad):
    '''
     Convert angle from radian to degree.
     
    '''
    return angle_rad/math.pi*180


def angleininterval(angle_degree):
    '''
    Convert input angle to be in the interval of [-180, 180] degree.

    '''
    angle_in = angle_degree % 360
    if angle_in>=180:
        angle_in = angle_in-360
    return angle_in

    
def get_angle_distance(xy_robot,xy_shape):
    '''
     Return the distance and angle between the robot and object
     with respect to x-axis and counter-clockwise orientation.
     
     << INPUT >>
     xy_robot : Robot location 
     xy_shape : Object loctaion 
     
     << OUTPUT >>
     angle: angle between the robot and the object
     distance: distance between the robot and the object
    '''
    distance = linalg.norm(xy_robot-xy_shape,2)
    angle = rad2degree(math.atan2(-xy_shape[1]+xy_robot[1],xy_shape[0]-xy_robot[0]))
    # steer angle for an integer 5deg, not 4.1deg
    print("angle", angle)
    return angle, distance

def pixel_to_cm(p):
    '''
    Since the robot movement cannot be measured by pixels, 
    we have to compute the relative measurement of the pixels to centimeter.
    '''
    const = 1220
    return p/const*160