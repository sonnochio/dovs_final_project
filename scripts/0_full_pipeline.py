from homography import perspective_transform
from feature_pipeline import detect_parking_bays
from digital_twin import draw_car_park
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Change the file name below to the car park image you want to process
file_name='assets/raw/large.jpg'

#This saves the transformed file under assets/warped 
perspective_transform(file_name='assets/raw/large.jpg')

#Detect parking bays and produce the number of rows and columns of the car park visible in the selected area 
image_path=f"assets/warped/{file_name[11:]}"
rows, cols, bays = detect_parking_bays(image_path=image_path)

#This will draw the digital twin of the car park
draw_car_park(rows=rows, cols=cols)
