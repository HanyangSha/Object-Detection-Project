#from re import X
import tensorflow as tf
#import matplotlib.pyplot as plt
import os
from PIL import Image
import gdown

#import argparse
import numpy as np
#from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
#from keras.layers.merge import add, concatenate
#from keras.models import Model
#import struct
import cv2
from copy import deepcopy
import functools

# Prepare data
DATA_ROOT = '/content/data'
os.makedirs(DATA_ROOT, exist_ok=True)

# model_url = 'https://drive.google.com/uc?id=19XKJWMKDfDlag2MR8ofjwvxhtr9BxqqN'
model_path = os.path.join(DATA_ROOT, 'yolo_weights.h5')
# gdown.download(model_url, model_path, True)
!wget -O /content/data/yolo_weights.h5 "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20%20Object%20Detection%20(Autonomous%20Vehicles)/yolo.h5"

labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]  

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.objness = objness
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes) # return index of max value of in classes 
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()] # probability score of the class of the greastest probability 
            
        return self.score

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3          

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

# compute intersection area/union area 
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

# scale and save the image into its target size 
def preprocess_input(image_pil, net_h, net_w):
    image = np.asarray(image_pil)
    new_h, new_w, _ = image.shape

    # determine the new size of the image while maintaining aspect ratio 
    if (float(net_w)/new_w) < (float(net_h)/new_h): # shrinking width more than height
        new_h = (new_h * net_w)/new_w # scale height
        new_w = net_w # change weight
    else:
        new_w = (new_w * net_h)/new_h
        new_h = net_h

    # resize the image to the new size
    #resized = cv2.resize(image[:,:,::-1]/255., (int(new_w), int(new_h)))
    resized = cv2.resize(image/255., (int(new_w), int(new_h)))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[int((net_h-new_h)//2):int((net_h+new_h)//2)-1, int((net_w-new_w)//2):int((net_w+new_w)//2), :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image

# takes the DarkNet output feature maps yolo_outputs as input
# returns all the predicted bounding boxes that have a higher objectness than the objectness threshold obj_thresh
def decode_netout(netout_, obj_thresh, anchors_, image_h, image_w, net_h, net_w):
    netout_all = deepcopy(netout_)

    # stores all valid boxes 
    boxes_all = []

    # iterating on bounding box dimensions 
    for i in range(len(netout_all)):
      netout = netout_all[i][0] # netout.shape = (S, S, 255)
      anchors = anchors_[i] # anchors is a list of different anchors for the ith dimension 

      grid_h, grid_w = netout.shape[:2] # slice: 0th index to 1st index 
      nb_box = 3
      netout = netout.reshape((grid_h, grid_w, nb_box, -1))
      # nb_class = netout.shape[-1] - 5

      boxes = []

      netout[..., :2]  = _sigmoid(netout[..., :2]) # applying sigmoid to box center offset
      netout[..., 4:]  = _sigmoid(netout[..., 4:]) # applying sigmoid to objectness score and class probabilities 
      netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:] # separating the 255 into 3 x 85 
      netout[..., 5:] *= netout[..., 5:] > obj_thresh # becomes 0 if obj_thresh is not met 

      # iterating over all grids 
      for i in range(grid_h*grid_w):
          row = i // grid_w # floor division 
          col = i % grid_w
          
          # iterating over bounding boxes 
          for b in range(nb_box):
              # element at index 4 is objectness score
              objectness = netout[row][col][b][4]
              # last elements are class probabilities
              classes = netout[row][col][b][5:]
              
              # if class probabilities for "person" less than object thresh 
              if(classes[0] <= obj_thresh): continue 
              
              # first 4 elements are x, y, w, and h
              x, y, w, h = netout[row][col][b][:4]

              # scaling hack  
              x = (col + x) / grid_w # center position, unit: image width
              y = (row + y) / grid_h # center position, unit: image height
              w = anchors[b][0] * np.exp(w) / net_w # unit: image width
              h = anchors[b][1] * np.exp(h) / net_h # unit: image height  
            
              # create box object 
              box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)

              boxes.append(box)

      boxes_all += boxes    
    
    # Correct boxes
    boxes_all = correct_yolo_boxes(boxes_all, image_h, image_w, net_h, net_w)
    
    return boxes_all

# random scaling stuff
def correct_yolo_boxes(boxes_, image_h, image_w, net_h, net_w):
    boxes = deepcopy(boxes_)

    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_w
        new_w = (image_w*net_h)/image_h
        
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

    return boxes

# removes all the bounding boxes that have a big (higher overlap than the nms_thresh) overlap with other better bounding boxes
def do_nms(boxes_, nms_thresh, obj_thresh):
    boxes = deepcopy(boxes_)
    if len(boxes) > 0:
        num_class = len(boxes[0].classes)
    else:
        return boxes

    # iterating on class     
    for c in range(num_class):
        # sorting in descending class probabilities for a specific class over all boxes
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])  

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                # intersection/union threshold hacking 
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh: # ignore the box at index_j, index_i has higher probability than index_j
                    boxes[index_j].classes[c] = 0

    new_boxes = []

    for box in boxes:
        label = -1
        for i in range(num_class):
            if box.classes[i] > obj_thresh:
                label = i
                box.label = label
                box.score = box.classes[i]
                new_boxes.append(box)  

    return new_boxes


from PIL import ImageDraw, ImageFont
import colorsys

def draw_boxes(image_, boxes, labels):
    image = image_.copy()
    image_w, image_h = image.size
    font = ImageFont.truetype(font='/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
                    size=np.floor(3e-2 * image_h + 0.5).astype('int32'))
    thickness = (image_w + image_h) // 300

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(labels), 1., 1.)
                  for x in range(len(labels))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.

    for i, box in reversed(list(enumerate(boxes))):
        c = box.get_label()
        predicted_class = labels[c]
        score = box.get_score()
        top, left, bottom, right = box.ymin, box.xmin, box.ymax, box.xmax

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        #label_size = draw.textsize(label)

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image_h, np.floor(bottom + 0.5).astype('int32'))
        right = min(image_w, np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
    return image  

def get_morph(img):
    # these lower and upper depend on the actual image
    lower = (100,130,130)
    upper = (180,200,200)
    thresh = cv2.inRange(img, lower, upper)

    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((5,5), np.uint8)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    return morph

def get_crosswalk(new_image):
    morph = get_morph(new_image)
    cntrs = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if (len(cntrs) == 2): # previous versions returns ret_img, contours, hierarchy
        cntrs = cntrs[0]
    else: # opencv3.1 returns ret_img, contours, hierarchy
        cntrs = cntrs[1]

    good_contours = [] # vector of vctors: first contours, then the points in each contour
    size_thresh = 200
    for c in cntrs:
        area = cv2.contourArea(c)
        if area > size_thresh:
            good_contours.append(c) 

    # combine good contours
    contours_combined = np.vstack(good_contours) # probably stacks all contours into 1 contour

    # get convex hull
    result = new_image.copy()
    hull = cv2.convexHull(contours_combined) # outer hull for all contours combined
    cv2.polylines(result, [hull], True, (0,0,255), 2)

    return result, hull

class Pair:
    def __init__(self, x, y):
        self.x = x
        self.y = y 

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        if (self.y == other.y): return self.x < other.x
        return self.y < other.y
        

def orientation(a,b,c): # a, b, c are all pairs
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

def graham(hull_):
    global p0

    # hull should be a vector of (x,y)
    hull = [] # change hull to vector as needed

    # turn everything into fucking pairs
    for i in range(len(hull_)):
      hull.append(Pair(hull_[i][0][0], hull_[i][0][1]))

    if (len(hull) == 1): return None

    hull.sort() # default sort 
    p0 = hull[0]

    hull = sorted(hull, key=functools.cmp_to_key(compare)) # sort on polar angle clockwise

    final = []
    final.append(p0)
    for i in range(len(hull)):
        while(len(final) > 1 and orientation(final[len(final)-2], final[len(final)-1], hull[i]) >= 0): # generates hull in clockwise direction
            final.pop(len(final)-1) # pops if its going counterclockwise (not optimal)
        final.append(hull[i])

    return final

def check_right(p, hull): # returns true if on the right
    for i in range(0,len(hull)-1): 
        cur = hull[i]
        next = hull[i+1]
        if (orientation(cur,next,p) >= 0): return False; # if counterclockwise or collinear, then its on the left or on the same line
    return True

def check_pedestrian(boxes, hull): # returns true if theres pedestrians within hull
    for box in boxes:
        a = check_right(Pair(box.xmin,box.ymin), hull)
        b = check_right(Pair(box.xmax,box.ymin), hull)
        c = check_right(Pair(box.xmin,box.ymax), hull)
        d = check_right(Pair(box.xmax,box.ymax), hull) 
        if (a or b or c or d): return True;
    return False

##################################

anchors = [[[116,90], [156,198], [373,326]], [[30,61], [62,45], [59,119]], [[10,13], [16,30], [33,23]]]
darknet = tf.keras.models.load_model(model_path)

ok = 0

# put into function 
def detect_image(image_pil, image_h, image_w, obj_thresh = 0.4, nms_thresh = 0.45, darknet=darknet, net_h=416, net_w=416, anchors=anchors, labels=labels):
  global ok

  new_image = preprocess_input(image_pil, net_h, net_w)
  yolo_outputs = darknet.predict(new_image)
  boxes = decode_netout(yolo_outputs, obj_thresh, anchors, image_h, image_w, net_h, net_w)

  if (len(boxes) == 0):
    if (ok == 0):
        print("Cars can go")
    ok = 1
    return new_image # nothing is drawn 

  boxes = do_nms(boxes, nms_thresh, obj_thresh)
  final_img = draw_boxes(image_pil, boxes, labels)
  final_img = cv2.cvtColor(np.asarray(final_img), cv2.COLOR_RGB2BGR)
  result, hull = get_crosswalk(final_img)

  hull = graham(hull) # sort the hulls in clockwise order
  boolb = check_pedestrian(boxes, hull) 
  if (boolb): 
    if ok == 1:
        print("Cars stop")
    ok = 0
  print(boolb)
  
  return result

def detect_video(video_path, output_path, obj_thresh = 0.4, nms_thresh = 0.45, darknet=darknet, net_h=416, net_w=416, anchors=anchors, labels=labels):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_FourCC = cv2.VideoWriter_fourcc(*'mp4v')
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

    num_frame = 0
    while vid.isOpened():
      ret, frame = vid.read()
      num_frame += 1
      print("=== Frame {} ===".format(num_frame))
      if ret:
          my_frame = frame
          x = Image.fromarray(cv2.cvtColor(my_frame, cv2.COLOR_BGR2RGB))

          image_w, image_h = x.size
          detected = detect_image(x, image_h, image_w)

          y = cv2.cvtColor(np.asarray(detected), cv2.COLOR_RGB2BGR)
          out.write(y)
      else:
          break
    vid.release()
    out.release()
    print("New video saved!")

video_path = '/content/data/video1.mp4'
output_path = '/content/data/video1_detected.mp4'
detect_video(video_path, output_path)

