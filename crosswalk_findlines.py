import cv2
import numpy as np
import functools
import math

frame = cv2.imread('/content/sample_data/pedestrian_crosswalk.jpg')
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

sensitivity = 50
lower_white = np.array([0,0,255-sensitivity])
upper_white = np.array([255,sensitivity,255])
mask = cv2.inRange(hsv, lower_white, upper_white)

kernel = np.ones((3,3), np.uint8)
morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
kernel = np.ones((6,6), np.uint8)
morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

edges = cv2.Canny(morph, 200, 400)

rho = 1  # distance precision in pixel, i.e. 1 pixel
angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
min_threshold = 5  # minimal of votes
line_segments = cv2.HoughLinesP(edges, rho, angle, min_threshold, np.array([]), minLineLength=5, maxLineGap=20)
  
def get_slope(a):
    x1, y1, x2, y2 = a[0]
    if (x2-x1 == 0):
        return -(1e9+9)
    return (y2-y1)/(x2-x1)

def cmp_slope(a, b):
    return get_slope(a) - get_slope(b)

line_segments = sorted(line_segments, key=functools.cmp_to_key(cmp_slope)) # sort in increasing slope

similar_slope = []
cur_slope = []

degdiff_threshold = math.pi/18

i = 0
while (i < len(line_segments)):
#for i in range(0, len(line_segments)):
    cur = line_segments[i]
    deg = math.atan(get_slope(cur))

    if (len(cur_slope) == 0): 
        if (i != len(line_segments) -1): 
            if (math.atan(get_slope(line_segments[i+1]) - deg <= degdiff_threshold)):
                cur_slope.append(cur)
                #cur_slope.append(line_segments[i+1])
            else:
                None # since there is not any lines with cur slope, just ignore 
        else: # ignore
            None
    else:
        if (deg - math.atan(get_slope(cur_slope[0])) <= degdiff_threshold):
            cur_slope.append(cur)
        else:
            similar_slope.append(cur_slope)
            cur_slope = []
            i -= 1
    
    i += 1

mxidx = 0
for i in range(1,len(similar_slope)):
  if (len(similar_slope[i]) > len(similar_slope[mxidx])):
    mxidx = i

opt = similar_slope[mxidx]
points = []

#print(len(opt))
for temp in opt:
  for x1, y1, x2, y2 in temp:
    cv2.line(hsv, (x1, y1), (x2, y2), (255, 255, 255), 5) #draw all lines in the largest set of lines with "same" slope

    p = np.polyfit((x1, x2), (y1, y2), 1)
    slope = p[0]
    intercept = p[1]
    for x in range(x1, x2+1): 
      y = slope * x + intercept
      points.append([x, y])
  
from numpy import unique
from numpy import where
from sklearn.cluster import Birch, KMeans, DBSCAN, MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot

# define the model
model = DBSCAN(eps = 30, min_samples = 2) #change based on necessity 
# fit the model
points = np.asarray(points)
model.fit(points)
# assign a cluster to each example
yhat = model.fit_predict(points)
# retrieve unique clusters
clusters = unique(yhat)
#print(len(clusters))

mx_idx = 0
for cluster in clusters: 
  i = where(yhat == cluster)
  if (len(i[0]) > len(i[mx_idx])):
    mx_idx = i

# create scatter plot for samples from each cluster
#for cluster in clusters:
  # get row indexes for samples with this cluster
row_ix = where(yhat == clusters[mx_idx])

#color = np.random.choice(range(256), size=3)
#c = ((int)(color[0]), (int)(color[1]), (int)(color[2]))
c = (0, 247, 12)

for i in range(points[row_ix, 0].shape[1]): 
  # draw the cluster with the maximum points
  cv2.circle(frame, ((int)(points[row_ix, 0][0][i]), (int)(points[row_ix, 1][0][i])), radius = (int)(2), color = c, thickness = (int)(-1))

class Pair:
    def __init__(self, x, y):
        self.x = x
        self.y = y 

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        if (self.y == other.y): return self.x < other.x
        return self.y < other.y
        

def orientation(a,b,c):
    v = a.x*(b.y-c.y)+b.x*(c.y-a.y)+c.x*(a.y-b.y) # simplification of the cross product, moving from a to b, c is arbitrary 
    if (v < 0): return -1; # clockwise
    if (v > 0): return +1; # counter-clockwise
    return 0

p0 = Pair(-1,-1) # python is stupid

def compare(a, b): # sort like c++ pairs, too lazy to get minimum point
    o = orientation(p0,a,b)
    if (o == 0): # if angle is the same, closer one is chosen 
        return (p0.x-a.x)*(p0.x-a.x) + (p0.y-a.y)*(p0.y-a.y) - (p0.x-b.x)*(p0.x-b.x) + (p0.y-b.y)*(p0.y-b.y) 
    return o

def graham(hull):
    global p0

    # hull should be a vector of Pair(x,y)
    if (len(hull) == 1): return None

    hull.sort() # default sort 
    p0 = hull[0]

    hull = sorted(hull, key=functools.cmp_to_key(compare)) # sort on polar angle clockwise

    final = []
    for i in range(len(hull)):
        while(len(final) > 1 and orientation(final[len(final)-2], final[len(final)-1], hull[i]) >= 0):
            final.pop(len(final)-1)
        final.append(hull[i])

    return final

import copy

hull = []
for i in range(points[row_ix, 0].shape[1]):
  # add all points from largest cluster to convex hull
  hull.append(Pair((int)(points[row_ix, 0][0][i]), (int)(points[row_ix, 1][0][i])))
#print(len(hull))

'''
opt_points = []
for temp in opt:
  for x1, y1, x2, y2 in temp:
    opt_points.append([x1,y1])
    opt_points.append([x2,y2])
opt_points = np.asarray(opt_points)
yhat2 = model.fit_predict(opt_points)

hull = []
for i in range(len(opt_points)):
  if (yhat2[i] == clusters[mx_idx]):
    hull.append(Pair(opt_points[i][0], opt_points[i][1]))
print(len(hull))
'''

hull = graham(hull)

frame2 = copy.deepcopy(frame)

# draw the convex hull
hull2 = []
for p in hull:
  hull2.append([p.x, p.y])
hull2 = np.asarray(hull2)
hull2 = hull2.reshape((-1, 1, 2))

cv2.polylines(frame, np.int32([hull2]), True, (0,0,255), 2)
cv2.imshow(frame)
