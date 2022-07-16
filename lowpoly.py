import sys
import cv2 as cv
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Delaunay
from PIL import Image, ImageOps, ImageDraw

#1) Load Image-------------------------------------
img_name = 'beatles_-_abbey_road.jpg'
img = Image.open('beatles_-_abbey_road.jpg') #May help to use np.asarray
width, height = img.size
img_np = img.load()

#2) Determine probability that pixel gets point dropped-------------------------------------------------------

#a) Calculate the gradient of a pixel (based on some number of neighbors) [for now, we'll just say that |gradient| > x, but worth looking into whether Laplacian Zero crossing is greater than a threshold]
#b) Find Laplacian Zero crossings of 3x3 patches
#c) Plot zero-crossings whose strength are greater than a threshold
    #Note: This means-- normalize strength of zero-crossing such that it's between [0, 1]-- this is percent chance it gets a point plotted
    #If there isn't a zero-crossing, the gradient will be low so either have a fixed probability or use some factor of the gradient
#d) Put these probabilities into 2D array

ddepth = cv.CV_16S
kernel_size = 3
src = cv.imread(cv.samples.findFile(img_name), cv.IMREAD_COLOR)
src = cv.GaussianBlur(src, (3, 3), 0)
src_grey = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
dst = cv.Laplacian(src_grey, ddepth, ksize=kernel_size)
img_edges = cv.convertScaleAbs(dst)
chances = img_edges/255


n = 1000

#sample1=pd.DataFrame(data=chances).stack().sort_values(ascending=False)[0:n].sample(n//2).index[0:n//2]
#sample2=pd.DataFrame(data=chances).stack().sort_values(ascending=False)[0:n].sample(n//2).index[0:n//2]
#max_index = sample1.append(sample2)
max_index = pd.DataFrame(data=chances).stack().sort_values(ascending=False).index[0:n]
x = max_index.get_level_values(1)
y = max_index.get_level_values(0)

coords = [(i, j) for i, j in zip(x, y)]
coords_reverse = [(j, i) for i, j in zip(x, y)]
coord_list = [[i, j] for i, j in zip(x, y)]
triangulate_coords = [(i, height-j) for i, j in zip(x, y)]

chance_array = pd.DataFrame(data=chances)

#3) Create new PIL image and plot points---------------------------------------------------------

pic = Image.new(mode = "RGB", size=(width, height))
draw = ImageDraw.Draw(pic)
draw.point(coords)

#4)Delaunay Triangulation + Color Fill---------------------------------------------------------------------------------------

#Note: We may use a different file here to do Delaunay Triangulation, and we're gonna use the neural network outlined in that research paper (find on Notion)

tri = Delaunay(coords)
test = pd.DataFrame(data = coords)
#draw.polygon(tri.simplices)
new_coords = np.array(coords)
#i=0
for triangle in new_coords[tri.simplices]:
    row = triangle[1].tolist()[0]
    col = triangle[1].tolist()[1]
    color = img_np[col, row]
    draw.polygon([(triangle[0].tolist()[1], triangle[0].tolist()[0]), (triangle[1].tolist()[1], triangle[1].tolist()[0]), (triangle[2].tolist()[1], triangle[2].tolist()[0])], fill=color)
    #i += 1
pic.show()

