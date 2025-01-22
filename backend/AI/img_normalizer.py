import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
 
for (root,dirs,files) in os.walk('../img/animals', topdown=True):
    for file in files:
        if file is not None:
            img = cv.imread(root + '/' + file, cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, (64, 64))
            cv.imwrite(root + '/' + file, img)
            
