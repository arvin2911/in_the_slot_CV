# import libraries
import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from ultralytics import YOLO
from IPython.display import display, clear_output
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import dill
import ipywidgets as widgets
import sys
import os

# %matplotlib notebook # enables interactive mode in Jupyter Notebook

# Alternative for Python script to enable interactive plotting:
plt.ion()

# import calibration matrix
#TODO:add the script to the folder
mtx = np.load("Andrew_camera_matrix.npy")  
fx = mtx[0][0]
fy = mtx[1][1]
cx = mtx[0][2]
cy = mtx[1][2]

# Load the model
#TODO:add the model to the folder
model = YOLO('bestwpingpong.pt') 

cap = cv2.VideoCapture('Strike zone overlay/please_work/IMG_5296.MOV') #Read in pitch from folder
frame_list = []  #List where we will store each frame of the video
while(cap.isOpened()): 
      
# Capture frame-by-frame 
    ret, frame = cap.read() 
    if ret == True: 
    # Display the resulting frame 
#         cv2.imshow('Frame', frame) 
        frame_list.append(frame)
    # Press Q on keyboard to exit 
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
  

    else: 
        break

# the video capture object 
cap.release() 
  
# Closes all the frames 
cv2.destroyAllWindows() 

# Process the video and return the processed video's path
def process_video(video_path):
    # saved directory path
    DEST_DIR = "static/results"
    SAVED_DIR = "yoohoo"
    filename = os.path.basename(video_path)

    # Predict with model
    results = model.predict(video_path, save=True, project=DEST_DIR, name=SAVED_DIR, exist_ok=True)
    
    return os.path.join(DEST_DIR,SAVED_DIR,filename)   # return the video path



## DEFINE FUNCTIONS

# Run calibration sequence
def is_orange(image, x1, y1, x2, y2, lower_orange, upper_orange, color_threshold=0.5):
    roi = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    orange_area = np.sum(mask > 0)
    total_area = (x2 - x1) * (y2 - y1)
    if total_area == 0:
        return False
    orange_ratio = orange_area / total_area
    if orange_ratio > color_threshold:
        return True
    return False

# Converts YOLO (2D) coordinates to real-world (3D) coordinates
def convert_to_real_world_coordinates(x, y, d_pix, fx, fy, cx, cy, true_diameter):
    Z = (fx * true_diameter) / d_pix
    X = ((x - cx) * Z) / fx
    Y = ((y - cy) * Z) / fy

    return X, Y, Z

# Converts the depth estimation metric
def meters_to_inches(meters):
    return meters * 39.3701  

# Calculates the depth estimation of baseball/pingpongs in an image/frame of a video
def process_image(image_path):
    # Load the image with OpenCV
    current_frame = cv2.imread(image_path)
    
    # Ensure the image was loaded
    if current_frame is None:
        print(f"Failed to load image {image_path}")
        return
    
    # Predict using the model for baseball class (class_id 32)
    results = model.predict(current_frame,classes = [80], conf=0.6)
   
    # points list to return
    points = []

    # Iterate through the results
    for result in results:
        boxes = result.boxes

        if boxes.conf.size(0) > 0:
            # There are detections
            for i in range(boxes.xyxy.size(0)): # For each detection
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                
                # Calculate the diameter of the baseball (approximation)
                d_pix = ((x2 - x1) + (y2 - y1)) / 2
                
                # Filter for orange color
                lower_orange = np.array([10, 100, 100])
                upper_orange = np.array([25, 255, 255])
                if not is_orange(current_frame, x1, y1, x2, y2, lower_orange, upper_orange):
                    continue
                # Draw rectangle around the baseball
                cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                print(f"num detections: {len(boxes)}, x1: {x1} y1: {y1} x2: {x2} y2: {y2}")


                print(f"Diameter of baseball in pixels: {d_pix}")

                # get midpoint of the ball in the image (pixels)
                y = y1 + (y2 - y1) / 2
                x = x1 + (x2 - x1) / 2
                print(f"{x} {y}")
                print(x)
                # calculate real-world depth
                X, Y, Z = convert_to_real_world_coordinates(x, y, d_pix, fx, fy, cx, cy, 0.04)
                points.append((meters_to_inches(X), meters_to_inches(Y), meters_to_inches(Z)))

                print(f"Real-world coordinates. X:{X} Y:{Y} Z:{Z}")
                # cv2.putText(current_frame, f"Real world coordinates:", (x1, y1-500), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(current_frame, f"Point: {len(points) - 1}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                # cv2.putText(current_frame, f"X:{X}", (x1, y1-300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                # cv2.putText(current_frame, f"Y:{Y}", (x1, y1-200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                # cv2.putText(current_frame, f"Z:{Z}", (x1, y1-100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                # cv2.circle(current_frame, (int(x),int(y)), radius=5, color=(0, 0, 255), thickness=-1)
                        
    # Display the modified frame with bounding boxes
    cv2.imshow("Detected Baseball", current_frame)
    cv2.waitKey(0)  # Wait for a key press to close
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.imwrite('detected_balls.jpg',current_frame)
    
    return points

# Find the pingpong index that corresponds to our homeplate index by calculating their MSE distance to each other
def compute_mse_for_point(point_index, distance_matrix, expected_distances):
    # check for only 5 points
    if len(distance_matrix) != 5:
        raise ValueError("Distance matrix must have 5 points corresponding to the corners of the home plate.")

    actual_distances = np.sort(np.delete(distance_matrix[point_index], point_index))
    # print(actual_distances)

    mse = np.mean((actual_distances - expected_distances) ** 2)
    return mse


# Home Plate Configuration:
#           3           2
#           _____________
#           |           | 
#    (left) |           | (right)
#           |           |
#           4           1
#            \         /
#             \       /
#              \     /
#               \   /
#                \ /
#                 0 (Front of the plate)

# Determines the home plate corners of the given pingpong indexes
def determine_plate_corners(points, distance_matrix, front_tip_index):
    # check for only 5 points
    if len(distance_matrix) != 5:
        raise ValueError("Distance matrix must have 5 points corresponding to the corners of the home plate.")

    num_points = len(points)
    # Exclude front tip from possible back corners
    other_indices = [i for i in range(num_points) if i != front_tip_index]
    print(other_indices)

    # Calculate distances from front tip to other points
    distances_from_front = distance_matrix[front_tip_index, other_indices]
    print(distances_from_front)

    # Identify back corners as the two farthest points from the front tip
    back_corners_indices = np.argsort(-distances_from_front)[:2]  # Get indices of two largest distances
    back_corners = [other_indices[i] for i in back_corners_indices]

    # The remaining point is the side corners
    remaining_index = list(set(other_indices) - set(back_corners))

    # find the closest point
    if np.linalg.norm(points[remaining_index[0]]) < np.linalg.norm(points[remaining_index[1]]):
        closest_point_index = remaining_index[0]
        further_point_index = remaining_index[1]
    else:
        closest_point_index = remaining_index[1]
        further_point_index = remaining_index[0]
    print(closest_point_index)

    # find the closest side
    if (distance_matrix[closest_point_index, back_corners[0]] < distance_matrix[closest_point_index, back_corners[1]]):
        closest_side_back_point_index = back_corners[0]
        further_side_back_point_index = back_corners[1]
    else:
        closest_side_back_point_index = back_corners[1]
        further_side_back_point_index = back_corners[0]

    # determine handed-ness of the batter
    if points[closest_point_index][0] < points[further_point_index][0]:  # compare x values of the closest point and the further point, if the x value of the further point is greater, we are standing on the left side of the plate and the batter is right-handed
        print("Right-handed batter (batter is on the left side of plate)")
        return [front_tip_index, further_point_index, further_side_back_point_index, closest_side_back_point_index, closest_point_index]
    else:
        print("Left-handed batter (batter is on the right side of plate)")
        return [front_tip_index, closest_point_index, closest_side_back_point_index, further_side_back_point_index, further_point_index]

# Reorders the pingpong index to match the homeplate point configuration
def order_homeplate_points(frame, points, pingpong_id=80):
    # for detection in detections:
    #     # calculate real-world depth
    #     X, Y, Z = convert_to_real_world_coordinates(detection.x, detection.y, detection.width, fx, fy, cx, cy, 0.04)
    #     X, Y, Z = meters_to_inches(X), meters_to_inches(Y), meters_to_inches(Z)
    #     points.append([X, Y, Z])
    #     print(f"Real-world coordinates. X:{X} Y:{Y} Z:{Z}")

    #     # display real-world coordinates
    #     cv2.putText(frame, f"Real world coordinates for point {len(points) - 1}", (int(detection.x), int(detection.y - 400)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    #     cv2.putText(frame, f"X:{X}", (int(detection.x), int(detection.y - 300)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    #     cv2.putText(frame, f"Y:{Y}", (int(detection.x), int(detection.y - 200)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    #     cv2.putText(frame, f"Z:{Z}", (int(detection.x), int(detection.y - 100)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # # Show image
    # print("here")
    # cv2.imshow('Image Frame', frame)
    # cv2.waitKey(0) # waits until a key is pressed
    # cv2.destroyAllWindows() # destroys the window showing image
    # cv2.waitKey(1)

    # Find which points correspond to which corner of home plate
    # 1. Find the front tip by computing MSE for each point as the front tip
    distance_matrix = squareform(pdist(points, 'euclidean'))
    print(distance_matrix)
    expected_distances = np.array([11.8, 11.8, 17, 17]) #convert from inches to meters
    mse_scores = []
    for i in range(len(points)):
        mse = compute_mse_for_point(i, distance_matrix, expected_distances)
        mse_scores.append(mse)
        print(f"MSE for point {i} as back tip: {mse}")

    # Determine which point has the lowest MSE
    front_tip_index = np.argmin(mse_scores)
    print(f"Point {front_tip_index} is likely the front tip of the home plate based on MSE.")

    # 2. Find the other 4 corners
    corners = determine_plate_corners(points, distance_matrix, front_tip_index)
    print("Identified corners:", corners)
    return corners,[points[corners[0]], points[corners[1]], points[corners[2]], points[corners[3]], points[corners[4]]]
