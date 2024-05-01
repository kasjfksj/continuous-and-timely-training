import cv2
import numpy as np
import torch
cap=cv2.VideoCapture(0)
while True:
    success,img =cap.read()
    print(img.shape)