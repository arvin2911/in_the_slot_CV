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

# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

# Process the video and return the processed video's path
def process_video(video_path):
    # saved directory path
    DEST_DIR = "static/results"
    SAVED_DIR = "yoohoo"
    filename = os.path.basename(video_path)

    # Predict with model
    results = model.predict(video_path, save=True, project=DEST_DIR, name=SAVED_DIR, exist_ok=True)
    
    return os.path.join(DEST_DIR,SAVED_DIR,filename)   # return the video path
