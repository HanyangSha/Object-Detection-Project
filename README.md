# Object-Detection-Project

In this project, I use the YOLO object detection model to detect vehicles and pedestrians. I also found the pedestrian crosswalk region. Based on this architecture, I aim to create a mechanism to use real time video survillance of street intersections to make traffic signals more efficient: by allowing traffic as soon as all pedestrians have crossed (no more pedestrians are detected in the crosswalk region), the waiting time of vehicles can be decreased. 

Explanations of what each file does (somewhat in chronological order):

1. object_detection_base.py: basic usage of the YOLO model. This program can detect any of the 80 classes the YOLO model has learned.
2. check_stationary.py: tracks stationary vehicles and records how long they have been stationary.
3. crosswalk_findlines.py: uses morphologyEx, Canny, and HoughLinesP functions of the OpenCV library to identify the outlines of crosswalk, uses DBSCAN to try to eliminate any noise (things in the image that have the same color as crosswalks), and uses Graham Scan on the final set of points to bound the crosswalk region.
4. crosswalk_convexhull.py: incorporates (1) and (3) to determine if there are any people detected within crosswalk region. Prints to standard output a decision of whether traffic can resume. 
5. pedestrian_crosswalk_final.py: optimizes crosswalk computation by only computing it for the first frame, and identifies vehicles as well. Basically a refined version and clearer version of (4). 

Example of detection (one tracks the stationary time and the other tracks the crosswalk and pedestrians [outputs for traffic light decisions on standard output not shown]): 

https://user-images.githubusercontent.com/110644978/184735914-a041d4ea-c9ae-4220-a66c-e95872e58a3a.mp4

https://user-images.githubusercontent.com/110644978/184735935-1d13771d-b61f-4c63-a64e-11b90e68186c.mp4
