import cv2
import numpy as np
import os
import math

def writeDir(img, dir, filename):
    if not os.path.exists(dir):
        os.makedirs(dir)
    className = filename.split('_')[0] + "/"
    if not os.path.exists(dir + className):
        os.makedirs(dir + className)
    cv2.imwrite(dir + className + filename, img)

#convert RGB image to just blue channel
def red(img, dir, filename):
    rows, columns, channels = img.shape
    r = np.zeros((rows, columns, 3))
    r[:,:,2] = img[:,:,2]
    writeDir(r, dir, filename)

#convert RGB image to just green channel
def green(img, dir, filename):
    rows, columns, channels = img.shape
    g = np.zeros((rows, columns, 3))
    g[:,:,1] = img[:,:,1]
    writeDir(g, dir, filename)

#convert RGB image to just red channel
def blue(img, dir, filename):
    rows, columns, channels = img.shape
    b = np.zeros((rows, columns, 3))
    b[:,:,0] = img[:,:,0]
    writeDir(b, dir, filename)

def gray(img, dir, filename):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    writeDir(gray, dir, filename)
    
def thresh(img, dir, filename):
    ret, thresh1 = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)
    writeDir(thresh1, dir, filename)

#Hough Line Transform
def hough(img, dir, filename):
    dst = cv2.Canny(img, 50, 200, None, 3)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(dst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    writeDir(dst, dir, filename)

def medFilter(img, dir, filename):
    median = cv2.medianBlur(img, 5)
    writeDir(median, dir, filename)

def gaussBlur(img, dir, filename):
    blur = cv2.GaussianBlur(img, (5,5), 7)
    writeDir(blur, dir, filename)

dir = "Datasets/cdpDemo"
for filename in os.listdir(dir):
    if filename == 'dogs' or filename == 'cats' or filename == 'pandas':
        folder = filename
        for filename in os.listdir(dir + "/cdp/" + folder):
            img = cv2.imread(os.path.join(dir + "/cdp/" + folder, filename))
            blue(img, dir + "cdpBlue/", filename)
            red(img, dir + "cdpRed/", filename)
            green(img, dir + "cdpGreen/", filename)
            gray(img, dir + "cdpGray/", filename)
            thresh(img, dir + "cdpThresh/", filename)
            hough(img, dir + "cdpHough/", filename)
            medFilter(img, dir + "cdpMedFilter/", filename)
            gaussBlur(img, dir + "cdpGaussBlur/", filename)
        continue
    else:
        continue