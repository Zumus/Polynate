import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps

#1) Load Image-------------------------------------
img_name = 'beatles_-_abbey_road.jpg'
img = np.array(Image.open('beatles_-_abbey_road.jpg'))
width, height = img.size

#2) Determine probability that pixel gets point dropped-------------------------------------------------------

#a) Create two 2D arrays the same size as the img array
#b) Calculate the gradient of a pixel (based on some number of neighbors) [for now, we'll just say that |gradient| > x, but worth looking into whether Laplacian Zero crossing is greater than a threshold]
#c) Use this to find Laplacian Zero crossings of 3x3 patches
#d) Plot zero-crossings whose strength are greater than a threshold
    #Note: This means-- normalize strength of zero-crossing such that it's between [0, 1]-- this is percent chance it gets a point plotted
    #If there isn't a zero-crossing, the gradient will be low so either have a fixed probability or use some factor of the gradient
#e) Put these probabilities into 2D array

ddepth = cv.CV_16S
kernel_size = 3
src = cv.imread(cv.samples.findFile(img_name), cv.IMREAD_COLOR)
src = cv.GaussianBlur(src, (3, 3), 0)
src_grey = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
dst = cv.Laplacian(src_grey, ddepth, ksize=kernel_size)
img_edges = cv.convertScaleAbs(dst)
cv.imshow("", img_edges)
cv.waitKey(0)
chances = img_edges/255
n = 1000

chances_array = pd.DataFrame(data=chances)
chances_array


#3)Delaunay Triangulation---------------------------------------------------------------------------------------

#Note: We may use a different file here, and we're gonna use the neural network outlined in that research paper (find on Notion)



#4)Color in the triangles ----------------------------------------------------------------------------------------

#Note: We could get the coordinates of the triangles and find the color of the center pixel, then draw a polygon