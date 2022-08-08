import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image
import gdown

import argparse
import numpy as np
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
import struct
import cv2
from copy import deepcopy

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
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None, time = 0):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.objness = objness
        self.classes = classes

        self.label = -1
        self.score = -1

        self.time = time 

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes) # return index of max value of in classes 
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()] # probability score of the class of the greastest probability 
            
        return self.score

    '''
    def __hash__(self):
        return hash((self.xmin, self.ymin, self.xmax, self.ymax))

    def __eq__(self, other): 
        a = (self.xmin, self.ymin, self.xmax, self.ymax)
        b = (other.xmin, other.ymin, other.xmax, other.ymax)
        return a == b
    '''

    def __eq__(self, other):
        '''
        a = (self.xmin, self.ymin, self.xmax, self.ymax)
        b = (other.xmin, other.ymin, other.xmax, other.ymax)

        width = self.xmax - self.xmin
        height = self.ymax - self.ymin

        woffset = width * 0.1
        hoffset = height * 0.1

        offset = (woffset, hoffset, woffset, hoffset)

        lb = tuple(map(lambda x, y: x-y, a, offset))
        ub = tuple(map(lambda x, y: x+y, a, offset))

        for i in range(0,4):
            if (not(b[i] >= lb[i] and b[i] <= ub[i])):
                return False
        return True
        '''

        x_center = (self.xmin + self.xmax)/2
        y_center = (self.ymin + self.ymax)/2
        
        x_co = (other.xmin + other.xmax)/2
        y_co = (other.ymin + other.ymax)/2

        offset_threshold = 0.05

        width = offset_threshold*(self.xmax - self.xmin)
        height = offset_threshold*(self.ymax - self.ymin)

        if (not(x_center >= x_co - width and x_center <= x_co + width)):
            return False
        if (not(y_center >= y_co - height and y_center <= y_co + height)):
            return False
        return True


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
    new_image[int((net_h-new_h)//2):int((net_h+new_h)//2), int((net_w-new_w)//2):int((net_w+new_w)//2), :] = resized
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
              
              objects_of_interest = [1, 2, 3, 5, 7]

              # if class probabilities for vehicles are less than object thresh 
              flag = 0
              for x in objects_of_interest: 
                if classes[x] > obj_thresh: 
                    flag = 1
              if(flag == 0): continue 
              
              # first 4 elements are x, y, w, and h
              x, y, w, h = netout[row][col][b][:4]

              # scaling hack  
              x = (col + x) / grid_w # center position, unit: image width
              y = (row + y) / grid_h # center position, unit: image height
              w = anchors[b][0] * np.exp(w) / net_w # unit: image width
              h = anchors[b][1] * np.exp(h) / net_h # unit: image height  
            
              # create box object 
              box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes) # time is 0

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

def draw_boxes(image_, boxes, labels, fps):
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

    cnt_stationary = 0

    for i, box in reversed(list(enumerate(boxes))):
        c = box.get_label()
        predicted_class = labels[c]
        score = box.get_score()
        top, left, bottom, right = box.ymin, box.xmin, box.ymax, box.xmax

        label = ""

        '''
        if (box in active.keys()):
            label = '{} {:.2f} \nstationary {:.2f}'.format(predicted_class, score, (float)(active[box])/fps)
        else:
            label = '{} {:.2f}'.format(predicted_class, score)
        '''

        time_thresh = 0.1

        if (((float)(box.time)/fps) <= time_thresh): # moving
            label = '{} {:.2f}'.format(predicted_class, score)
        else: # stationary 
            label = '{} {:.2f} \nstationary {:.2f}'.format(predicted_class, score, (float)(box.time)/fps)
            cnt_stationary += 1

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        # do some stuff with image bounds
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image_h, np.floor(bottom + 0.5).astype('int32'))
        right = min(image_w, np.floor(right + 0.5).astype('int32'))

        print(label, (left, top), (right, bottom))

        # check if theres enough space to display the label above the box
        # if not, display it inside the top bound
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

    print("Number of vehicles stationary: ", cnt_stationary)
    return image  

anchors = [[[116,90], [156,198], [373,326]], [[30,61], [62,45], [59,119]], [[10,13], [16,30], [33,23]]]
darknet = tf.keras.models.load_model(model_path)

#active = {} # only contains already stationary cars 
prev_boxes = []

# put into function 
def detect_image(image_pil, image_h, image_w, obj_thresh = 0.4, nms_thresh = 0.45, darknet=darknet, net_h=416, net_w=416, anchors=anchors, labels=labels):
  global prev_boxes

  new_image = preprocess_input(image_pil, net_h, net_w)
  yolo_outputs = darknet.predict(new_image)
  boxes = decode_netout(yolo_outputs, obj_thresh, anchors, image_h, image_w, net_h, net_w)
  boxes = do_nms(boxes, nms_thresh, obj_thresh) # contains both moving and stationary 

  final_boxes = []

  # everything in boxes gets added to final_boxes
  for x in boxes:
    flag = 0
    for y in prev_boxes: 
        if (x == y): # calls __eq__
            a = deepcopy(y) # copy y, y has the time 
            a.time += 1
            final_boxes.append(a) # existed in the same location, so increase stationary time
            flag = 1
    if (flag == 0): # first time
        a = deepcopy(x)
        final_boxes.append(a) 

  prev_boxes = final_boxes

  final_img = draw_boxes(image_pil, final_boxes, labels, 30)
  return final_img


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

video_path = '/content/data/video4.mp4'
output_path = '/content/data/video1_detected.mp4'
detect_video(video_path, output_path)

