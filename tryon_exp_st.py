import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from matplotlib import pyplot as plt
import math
import warnings
import os
from pathlib import Path
import streamlit as st

warnings.filterwarnings('ignore')

st.title('Shirt size estimator')

file = st.file_uploader("Upload your photo", type=["jpg", "jpeg"])

if file is not None:

    # Initialize Mediapipe Pose.
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.75)

    # Initialize Mediapipe drawing utilities.
    mp_drawing = mp.solutions.drawing_utils

    # Load your image.

    # result_dict = {'name':[], 'predicted_size' : []}

    # for file in os.listdir(r'dec_team'):
    # name, ext = os.path.splitext(file)
    #result_dict['name'].append(name)
    # image_path_str = file
    # image_path = Path(image_path_str)
    image = np.array(Image.open(file).convert("RGB"))
    # image = cv2.resize(image, (1280, 960))
    image_original = Image.fromarray(image)
    #image_disp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image, (512, 512))
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

    # Process the image and find the pose landmarks.
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Draw pose landmarks on the image.
        #mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
        # Extract landmark coordinates.
        landmarks = results.pose_landmarks.landmark
    
        # Shoulders: Left (11), Right (12)
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    
        # Hips: Left (23), Right (24)
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    
        # Convert relative coordinates to pixel values.
        h, w, _ = image_rgb.shape
        left_shoulder_coords = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        right_shoulder_coords = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        left_hip_coords = (int(left_hip.x * w), int(left_hip.y * h))
        right_hip_coords = (int(right_hip.x * w), int(right_hip.y * h))
    
        # Calculate bounding box coordinates.
        x_min = min(left_shoulder_coords[0], right_shoulder_coords[0], left_hip_coords[0], right_hip_coords[0])
        x_max = max(left_shoulder_coords[0], right_shoulder_coords[0], left_hip_coords[0], right_hip_coords[0])
        y_min = min(left_shoulder_coords[1], right_shoulder_coords[1], left_hip_coords[1], right_hip_coords[1])
        y_max = max(left_shoulder_coords[1], right_shoulder_coords[1], left_hip_coords[1], right_hip_coords[1])
    
        left_shoulder_x = left_shoulder_coords[0]
        left_shoulder_y = left_shoulder_coords[1]
    
        right_shoulder_x = right_shoulder_coords[0]
        right_shoulder_y = right_shoulder_coords[1]
    
        # left_hip_x = left_hip_coords[0]
        # left_hip_y = left_hip_coords[1]
    
        # right_hip_x = right_hip_coords[0]
        # right_hip_y = right_hip_coords[1]
    
        # avg_y_shoulder = int((left_shoulder_y + right_shoulder_y)/2)
        
        print(f"Left dimensions: {(left_shoulder_x, left_shoulder_y)}")
        print(f"Right dimensions: {(right_shoulder_x, right_shoulder_y)}")
        
        pix_dist = math.sqrt(((right_shoulder_x - left_shoulder_x)**2) + ((right_shoulder_y - left_shoulder_y)**2))
        # pix_dist_hip = math.sqrt(((right_hip_x - left_hip_x)**2) + ((right_hip_y - left_hip_y)**2))
        # avg_pix_dist = (pix_dist + pix_dist_hip)/2
        #pix_dist = math.sqrt(((right_shoulder_x - left_shoulder_x)**2))
        img_dst_feet = 6
    
        # act_dist = 37.25
        act_dist = (pix_dist/2.7161895533695346)*(img_dst_feet/6)
        #result_dict['predicted_size'].append(round(act_dist, 2))
    
        # pixs_cm_shoulder =  2.7161895533695346
        # pixs_cm_hip = 1.450658371662421
        # #pixs_cm = (pixs_cm_shoulder + pixs_cm_hip)/2
    
    # print(f"Shirt size: {round(avg_pix_dist/2, 2)} cm")
        
        # print(f"Shirt Size: {round(act_dist, 2)}")
        
        #cv2.line(image_rgb, (left_shoulder_x, left_shoulder_y), (right_shoulder_x, right_shoulder_y), (0, 255, 0), 3)
    
        img_disp = cv2.resize(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB), (image_original.size))
        
        # cv2.putText(img_disp, f"Shirt Size = {round(act_dist, 2)}", (225, 225), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 200, 127), 3, cv2.LINE_AA)
        text_content = f"Shirt Size: {round(act_dist, 2)}"
        font_size = "20px"
        color = "blue"
        html_content = f'<p style="font-size:{font_size}; color:{color};">{text_content}</p>'
        st.markdown(html_content, unsafe_allow_html=True)
        #st.text(caption)
        st.image(image_original, caption= "model's image", use_column_width= True)
    else:
        text_content = "Unable to detect Human in Image. Please use a different image."
        font_size = "20px"
        color = "red"
        html_content = f'<p style="font-size:{font_size}; color:{color};">{text_content}</p>'
        st.markdown(html_content, unsafe_allow_html=True)
