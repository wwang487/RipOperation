# -*- coding: utf-8 -*-
"""
Created on Wed May  4 09:00:27 2022

@author: bbean
"""

import cv2
import numpy as np
import math

def EqualHist(gray_img):
    return cv2.equalizeHist(gray_img)

def EqualHistForRGBImg(rgb_img):
    r, g, b = cv2.split(rgb_img)
    return cv2.merge([EqualHist(r), EqualHist(g), EqualHist(b)])

def Clahe(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    dst = clahe.apply(gray_img)
    return dst

def ClaheForRGBImg(rgb_img):
    r, g, b = cv2.split(rgb_img)
    return cv2.merge([Clahe(r), Clahe(g), Clahe(b)])

def LinearTransform(gray_img, a):
    O = float(a) * gray_img
    O[O>255] = 255 
    O = np.round(O)
    O = O.astype(np.uint8)
    return O

def LinearTransformForRGBImg(rgb_img, a):
    r, g, b = cv2.split(rgb_img)
    return cv2.merge([LinearTransform(r, a), LinearTransform(g, a), LinearTransform(b, a)])

def GammaTransform(gray_img, gamma):
    fI = gray_img/255.0
    O = np.power(fI, gamma)
    O1 = O*255
    O1[O1>255] = 255
    return O1.astype(np.uint8)

def GammaTransformForRGBImg(rgb_img, gamma):
    r, g, b = cv2.split(rgb_img)
    return cv2.merge([GammaTransform(r, gamma), GammaTransform(g, gamma), GammaTransform(b, gamma)])

def ComputeHistogram(gray_img):
    return cv2.calcHist([gray_img], [0], None, [256], [0, 256])

def Normalize(gray_img):
    dst = np.zeros_like(gray_img)
    return cv2.normalize(gray_img, dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

def NormalizeRGBImg(rgb_img):
    r, g, b = cv2.split(rgb_img)
    return cv2.merge([Normalize(r), Normalize(g), Normalize(b)])
