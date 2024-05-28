# import libraries
import modules

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
import mpld3
import plotly.graph_objects as go


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
def process_image(image_path, model, fx, fy, cx, cy):
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
    # cv2.imshow("Detected Baseball", current_frame)
    # cv2.waitKey(0)  # Wait for a key press to close
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # cv2.imwrite('detected_balls.jpg',current_frame)
    
    return points

# Processes the baseball image in a given image/frame in a video
def process_image_baseball(image_path, model, fx, fy, cx, cy):
    # Load the image with OpenCV
    current_frame = cv2.imread(image_path)
    
    # Ensure the image was loaded
    if current_frame is None:
        print(f"Failed to load image {image_path}")
        return
    
    # Predict using the model for baseball class (class_id 32)
    results = model.predict(current_frame,classes = [32], conf=0.1)
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
                print('diameter pixel ball = ')
                print(d_pix)
                # Filter for size
#                 lower_orange = np.array([0, 0, 200])
#                 upper_orange = np.array([180, 50, 255])
                # if d_pix < 30:
                #     continue
                # Draw rectangle around the baseball
                print('diameter pixel ball < 30')
                cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                print('after cv2 draw rect')
                print(f"num detections: {len(boxes)}, x1: {x1} y1: {y1} x2: {x2} y2: {y2}")


                print(f"Diameter of baseball in pixels: {d_pix}")

                # get midpoint of the ball in the image (pixels)
                y = y1 + (y2 - y1) / 2
                x = x1 + (x2 - x1) / 2
                print(f"{x} {y}")
                print(x)
                # calculate real-world depth
                X, Y, Z = convert_to_real_world_coordinates(x, y, d_pix, fx, fy, cx, cy, 0.075)
                points.append((meters_to_inches(X), meters_to_inches(Y), meters_to_inches(Z)))
                print(f"Real-world coordinates. X:{X} Y:{Y} Z:{Z}")
                # cv2.putText(current_frame, f"Real world coordinates:", (x1, y1-500), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(current_frame, f"Point: {len(points) - 1}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                # cv2.putText(current_frame, f"X:{X}", (x1, y1-300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                # cv2.putText(current_frame, f"Y:{Y}", (x1, y1-200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                # cv2.putText(current_frame, f"Z:{Z}", (x1, y1-100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                # cv2.circle(current_frame, (int(x),int(y)), radius=5, color=(0, 0, 255), thickness=-1)
                        
    # Display the modified frame with bounding boxes
    # cv2.imshow("Detected Baseball", current_frame)
    # cv2.waitKey(0)  # Wait for a key press to close
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
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


# Build the strike zone
def build_zone(world_coordinate_list):
    #The world coordinate list should be ordered as follows: The 1st entry should be the position of the ball that is at the back (the point) of the plate, from there, 
    #rotate counterclockwise until every ball on the plate is accounted for. The sixth entry will be the lower bound for the zone, the seventh entry will be the upper bound of the zone
    #We're also going to average over the coordinates of the corners of the zone to mitigate errors in strike zone dimensions
# For the strike zone, the width of the plate from this perspective is the x coordinate (increasing from left to right), the length is the z coordinate (increasing from top to bottom of the screen
#and the y coordinate is the height of the zone coming out of the screen. REMEMBER THIS
     #/\
    #|  |
    #____
    #Front of the zone
    #Front plane of the zone length in x coordinate
    front_plane_length_x = world_coordinate_list[2][0] - world_coordinate_list[3][0] 
    back_plane_length_x = world_coordinate_list[1][0] - world_coordinate_list[4][0] 

    x_length_front = ( front_plane_length_x + back_plane_length_x )/2 #average of the 2 lengths

    #z length of the front zone. This is the depth of the plate

    right_plane_length_z = (world_coordinate_list[2][2]) - (world_coordinate_list[1][2]) 
    
    left_plane_length_z = world_coordinate_list[4][2] - world_coordinate_list[3][2] 
    print(right_plane_length_z)
    print(left_plane_length_z)
    z_length_front = max(np.abs(left_plane_length_z),np.abs(right_plane_length_z)) #average of the 2 lengths

    y_bottom =  world_coordinate_list[5][1] #Height of the bottom of the zone
    y_top = world_coordinate_list[6][1] #height of the top of the zone
    y_length = y_top - y_bottom
    #Now for the back of the zone where we have curvature. We want to return the slope of the rear of the zone so we can use that for ball and strike checking later
    back_zone_left_edge_slope = (world_coordinate_list[0][2] - world_coordinate_list[4][2] )/( world_coordinate_list[0][0] - world_coordinate_list[4][0] ) #change in z over change in x babay

    back_zone_right_edge_slope = (world_coordinate_list[0][2] - world_coordinate_list[1][2] )/( world_coordinate_list[0][0] - world_coordinate_list[1][0] ) #change in z over change in x babay
    #Also want x intercepts of both of these lines

    intercept_left = world_coordinate_list[0][2] - back_zone_left_edge_slope*world_coordinate_list[0][0] #calculate z intercept of the left slope of the zone
    intercept_right = world_coordinate_list[0][2] - back_zone_right_edge_slope*world_coordinate_list[0][0] #calculate the z intercept of the right slope
    #We also want the y dimension length of the back of the zone. So the tip of the plate to the back plane of the zone
    z_length_back = max(  (np.abs(world_coordinate_list[0][2] - world_coordinate_list[4][2]) ), (np.abs(world_coordinate_list[0][2] - world_coordinate_list[1][2]) ) ) #average over 2 possible lengths for the 
                                                                                                                                                    #length of the triangular part of the zone
    #Going to return these lengths as a dictionary

    zone_dim_dict = {
        'x_length_front': x_length_front,
        'y_bottom': y_bottom,
        'y_top': y_top,
        'y_length': y_length,
        'z_length_front': z_length_front,
        'z_length_back': z_length_back,
        'back_zone_left_slope': back_zone_left_edge_slope,
        'back_zone_right_slope': back_zone_right_edge_slope, 
        'intercept_left': intercept_left,
        'intercept_right': intercept_right
    }

    return zone_dim_dict # return a dictionary of the bounding edges of the zone

# Check if baseball pitch is a strike 
def check_strike(ball_position, zone_bounds):
    #ball position: tuple of the world coordinates of the ball
    #Zone bounds: list of tuples that contains the center positions of each of the balls used to mark the edges of the strike zone.
    #The 1st entry should be the position of the ball that is at the back (the point) of the plate, from there, 
    #rotate counter clockwise until every ball on the plate is accounted for. The sixth entry will be the lower bound for the zone, the seventh entry will be the upper bound of the zone
    #We're also going to average over the coordinates of the corners of the zone to mitigate errors in strike zone dimensions
    
    zone_dims_dict = build_zone(zone_bounds)
    
    z_check_front = ball_position[2] >= zone_bounds[3][2] and ball_position[2] <= zone_bounds[3][2] + zone_dims_dict.get('z_length_front') #Check if ball is outside the front of zone to start 
    if z_check_front == True: #Ball is in the front rectangular region of the zone
        x_check_front = ball_position[0] >= zone_bounds[3][0] and ball_position[0] <= zone_bounds[3][0] + zone_dims_dict.get('x_length_front') #Check if ball is in x range of the front of the zone (width of the zone)
        y_check_front = ball_position[1] >= zone_dims_dict.get('y_bottom') and ball_position[1] <= zone_dims_dict.get('y_top') #check if ball is in the height of zone 
        if x_check_front == True and y_check_front == True:
            strike = 1
            return strike

    z_check_back = ball_position[2] >= zone_bounds[4][2] and ball_position[2] <= zone_bounds[4][2] + zone_dims_dict.get('z_length_back') #check if ball z coordinate is in the back portion of the zone (triangular part)
    # print(z_check_back)
    if z_check_back == True:
        #Using the x intercepts of the lines calculated in zone_dims 
        x_check_back = ball_position[0] >= ( ball_position[2] - zone_dims_dict.get('intercept_left') ) / (zone_dims_dict.get('back_zone_left_slope')) \
                        and ball_position[0] <= ( ball_position[2] - zone_dims_dict.get('intercept_right') ) / (zone_dims_dict.get('back_zone_right_slope'))
        
        y_check_back = ball_position[1] >= zone_dims_dict.get('y_bottom') and ball_position[1] <= zone_dims_dict.get('y_top')#check if ball is in the y range of the zone (the height)
        # print(y_check_back)
        if x_check_back == True and y_check_back == True:
            strike = 1
            return strike
    strike = 0
    return strike

def draw_pentagonal_prism(zone_bounds, zone_dims_dict, center=(0, 0, 0)):
    vertices_base = np.array([
        [zone_bounds[i][0], zone_bounds[i][2], zone_bounds[i][1]] for i in range(len(zone_bounds) - 2)
    ])

    height = zone_dims_dict.get('y_length')
    vertices_top = vertices_base + np.array([0, 0, -height])
    vertices = np.vstack([vertices_base, vertices_top])

    # Extract x, y, z coordinates for each vertex
    x, y, z = np.array(vertices).T

    i = []
    j = []
    k = []

    # Create the faces of the prism
    for m in range(len(vertices_base)):
        i.append(m)
        j.append((m + 1) % len(vertices_base))
        k.append(len(vertices_base) + (m + 1) % len(vertices_base))

        i.append(m)
        j.append(len(vertices_base) + (m + 1) % len(vertices_base))
        k.append(len(vertices_base) + m)

    # Top face
    i.extend(range(len(vertices_base)))
    j.extend([(m + 1) % len(vertices_base) for m in range(len(vertices_base))])
    k.extend([len(vertices_base)] * len(vertices_base))

    # Bottom face
    i.extend([len(vertices_base) + m for m in range(len(vertices_base))])
    j.extend([len(vertices_base) + (m + 1) % len(vertices_base) for m in range(len(vertices_base))])
    k.extend([2 * len(vertices_base) - 1] * len(vertices_base))

    prism_trace = go.Mesh3d(x=x, y=y, z=z, color='cyan', opacity=0.6, i=i, j=j, k=k, name='Prism')

    return prism_trace
## MAIN EXECUTION BLOCK

def generate_result(video_path, model_path='bestwpingpong.pt', calibration_matrix_path="Andrew_camera_matrix.npy"):
    # Load the calibration matrix
    mtx = np.load(calibration_matrix_path)  
    fx = mtx[0][0]
    fy = mtx[1][1]
    cx = mtx[0][2]
    cy = mtx[1][2]

    # Load the model
    model = YOLO(model_path) 

    cap = cv2.VideoCapture(video_path) #Read in pitch from folder
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

    # cap.release() 
    # cv2.destroyAllWindows() 

    # Load in video and save an initial frame to be used in process_image
    test_image = frame_list[40]
    image_path = "static/results/validation_image.png"
    cv2.imwrite(image_path, test_image)

    points = process_image(image_path, model, fx, fy, cx, cy)

    # Reorder corner points of the strike zone based on homeplate
    corners, point_order = (order_homeplate_points(test_image, points))

    # Append on a zone height
    point_order.append((0,18,0)) #bottom border of the zone
    point_order.append((0,38,0)) #top of the zone

    expected_distances = np.array([11.8, 17, 8.5]) #convert from inches to meters
    zone_dims_dict = build_zone(point_order)
    print("start")
    baseball_position = process_image_baseball(image_path, model, fx, fy, cx, cy)[0]
    result = [check_strike(baseball_position,point_order)]
    print("1")

    ball_positions = [baseball_position]
    print("2")

    center = (point_order[0][0] ,point_order[0][2] + zone_dims_dict.get('z_length_back'),point_order[0][1] + zone_dims_dict.get('y_length')*.5  )
    print(center)
    prism_trace = draw_pentagonal_prism(zone_bounds=point_order, zone_dims_dict=zone_dims_dict, center=center)
    fig = go.Figure(data=[prism_trace])

    # Update layout
    fig.update_layout(scene=dict(
        xaxis=dict(title='x', range=[-15, 5]),
        yaxis=dict(title='y', range=[70, 100]),
        zaxis=dict(title='z', range=[-2 * zone_dims_dict['y_length'], 2 * zone_dims_dict['y_length']])
    ))

    for i in range(len(ball_positions)):
        color = 'red' if result[i] == 1 else 'blue'
        fig.add_trace(go.Scatter3d(x=[ball_positions[i][0]], y=[ball_positions[i][2]], z=[ball_positions[i][1]], mode='markers', marker=dict(color=color, size=10)))

    # Add legend
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(color='red', size=10), name='Strike'))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(color='blue', size=10), name='Ball'))

    # Save the interactive plot as an HTML file
    plot_path = 'static/results/interactive_plot.html'
    fig.write_html(plot_path)

    # Return the path to the generated HTML plot
    return plot_path


# Process the video and return the processed video's path
# def process_video(video_path):
#     # saved directory path
#     DEST_DIR = "static/results"
#     SAVED_DIR = "yoohoo"
#     filename = os.path.basename(video_path)

#     # Predict with model
#     results = model.predict(video_path, save=True, project=DEST_DIR, name=SAVED_DIR, exist_ok=True)
    
#     return os.path.join(DEST_DIR,SAVED_DIR,filename)   # return the video path




