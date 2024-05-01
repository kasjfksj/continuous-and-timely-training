import cv2
import numpy as np
import torch
from model import model
cap=cv2.VideoCapture(0)
a = model()
while True:
    success,img =cap.read()