
import matplotlib.pyplot as plt
import numpy as np
import cv2
from time import sleep
import time

import numpy as np
import math

# https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a

class LineFinder:
    def __init__(self, img):
        self.img = img
        m,n,c = img.shape
        self.shape = m,n

        self.orange_map = np.zeros((m,n), dtype=np.uint8) # stores the location all orange pixels
        self.center_map = np.zeros((m,n), dtype=np.uint8) # stores the locations of the centers of objects

        self.id_counter= 0


    def find_orange_lines(self):
        '''
        perform the full pipeline of isolating cones and then plotting a trend line over them
        '''
        self.create_orange_map() # isolates orange pixels
        ret, labels = self.find_objects() # finds connected components
        self.find_centers(ret, labels) # isolates the center of each connected component
        self.drawLines() # uses hough line transform and uses the highest scoring lines to mark the directions

    def drawLines(self):
        '''
        find two lines using the hough transform
        and take the two lines with the most acculumated hough transform counts
        only considers nearly vertical lines
        
        plot the top two lines over the image
        '''
        # look for vertical lines
        lines = cv2.HoughLines(self.center_map, 1, np.pi/180, 4, min_theta = 5*np.pi/6, max_theta=np.pi)
        lines2 = cv2.HoughLines(self.center_map, 1, np.pi/180, 4, min_theta = 0, max_theta = np.pi/6)

        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        img = self.img
        if type(lines) == type(None) or type(lines2) == type(None):
            print("no lines")
            return

        # plot the most likely lines
        rho1, theta1 = lines[0,0,:]
        rho2, theta2 = lines2[0,0,:]
        for rho,theta in [(rho1, theta1), (rho2, theta2)]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            linelen=3000
            x1 = int(x0 + linelen *(-b))
            y1 = int(y0 + linelen*(a))
            x2 = int(x0 - linelen*(-b))
            y2 = int(y0 - linelen*a)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imshow("final", img)
            cv2.waitKey(0)
        cv2.imwrite("myanswer.png", img)



    def find_object_center(self, obj_map):
        '''
        finds the center of an individual object
        arg: obj_map, mxn binary pixel map containing the location of a single object

        returns:
        yc : the y axis center
        xc : the x axis center of the object
        '''
        print("timing find object centers")
        t1 = time.time()
        X2 = obj_map.sum(axis=0)
        Y2 = obj_map.sum(axis=1)

        X = np.zeros(X2.shape)
        Y = np.zeros(Y2.shape)

        for i in range(len(X)):
            X[i] = X2[i]*i
            Y[i] = Y2[i]*i
        print(np.sum(X), np.sum(X2), np.sum(Y), np.sum(Y2))
        xc = int(np.sum(X) / np.sum(X2))
        yc = int(np.sum(Y) / np.sum(Y2))
        print(time.time()-t1)

        return yc,xc


    def find_centers(self, ret, labels):
        '''
        iterate over each connected component and extract the center
        args:
        ret : the number of distinct object
        labels : a pixel map of the segmented object

        returns:
            nothing, but sets center_map to 1 at the position of the calculated center
        '''
        for label in range(1, ret):
            mask = np.array(labels, dtype=np.uint8)
            mask[labels==label] = 255
            mask[labels != label] = 0
            # cv2.imshow("mask2", mask)
            y,x = self.find_object_center(mask)
            S = 1
            self.center_map[(y-S):(y+S),(x-S):(x+S)] = 1
            # cv2.imshow("window", self.center_map*254)
        
        
    def create_orange_map(self):
        '''
        extract orange pixels
        and then applies a blur filter so that noisy readings don't create multiple objects out of one
        '''
        print("creating orange map")
        m,n = self.shape
        orange = np.ones((m,n,3))
        orange[:,:,0] = 30/255.0
        orange[:, :, 1] = 30/255.0
        orange[:, :, 2] = 190/255.0
        img = self.img / 255.0

        map = (orange-img)
        map = np.abs(map).sum(axis=2)
        map = map < 0.25
        map = map.astype(np.uint8) * 200
        kernel = np.ones((5,5), np.float32)/25
        map = cv2.filter2D(map, -1, kernel)
        self.orange_map = map
    
    
    def find_objects(self):
        ''''
        isolates connected components of the image
        plots the location of each image if desired.
        returns:
        ret - the number of objects
        labels - the pixel maps corresponding to each label
        '''
        ret, labels = cv2.connectedComponents(self.orange_map)
        # print('num objects', ret)
        # for label in range(1,ret):
        #     mask = np.array(labels, dtype=np.uint8)
        #     mask[labels==label] = 255
        #     cv2.imshow("mask", mask)
        return ret, labels

src = cv2.imread("cones.png")
p = LineFinder(src)
p.find_orange_lines()

cv2.waitKey(0)
cv2.destroyAllWindows()