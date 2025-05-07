# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 02:05:28 2020

@author: bbean
"""
import os
import cv2
import numpy as np
from PIL import Image
import argparse
#from concurrent.futures import ProcessPoolExecutor

def CheckIfFolderExist(filepath):
    if os.path.exists(f'{filepath}'):
        pass
    else:
        os.mkdir(f'{filepath}') 


def zmMinFilterGray(src, r=7):
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))


def guidedfilter(I, p, r, eps):
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def Defog(m, r, eps, w, maxV1):                

    V1 = np.min(m, 2)                           
    Dark_Channel = zmMinFilterGray(V1, 7)

    V1 = guidedfilter(V1, Dark_Channel, r, eps)  
    bins = 2000
    ht = np.histogram(V1, bins)                  
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
    V10 = np.minimum(V1 * w, maxV1)               
    return V10, A, Dark_Channel


def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    Mask_img, A, D = Defog(m, r, eps, w, maxV1)             

    for k in range(3):
        Y[:,:,k] = (m[:,:,k] - Mask_img)/(1-Mask_img/A)  
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))      
    return Y, Mask_img, D

def unevenLightCompensate(img, blockSize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (3, 3), 0)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    return dst

def create_res(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    res_ = cv2.resize(image,None,fx=0.6, fy=0.6, interpolation = cv2.INTER_CUBIC)
    mask = cv2.resize(mask,None,fx=0.6, fy=0.6, interpolation = cv2.INTER_CUBIC)
    dst = cv2.inpaint(res_, mask, 10, cv2.INPAINT_TELEA)
    return dst

def equal_hist(img):
    b, g, r = cv2.split(img)
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)
    enhanced_img = cv2.merge([b, g, r])
    return enhanced_img

def group_dehaze_processing(input_dir, suffix, output_dir, dests):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            filepath = os.path.join(root, file)
            # Here, split the path to get the suffix of file
            filesuffix = os.path.splitext(filepath)[1][1:]
            # If suffix matches, read the image.
            if filesuffix in suffix:
                img = cv2.imread(filepath)
               
                Y,M,D = deHaze(img / 255)
                Y1,M1,D1 = deHaze(img / 255, bGamma=True)

                ED = equal_hist(np.uint8(Y * 255))
                
                cv2.imwrite(output_dir + '/' + dests[0] + '/' + file, Y * 255)
                cv2.imwrite(output_dir + '/' + dests[1] + '/' + file, M * 255)
                cv2.imwrite(output_dir + '/' + dests[2] + '/' + file, D * 255)
                cv2.imwrite(output_dir + '/' + dests[3] + '/' + file, Y1 * 255)
                cv2.imwrite(output_dir + '/' + dests[4] + '/' + file, ED)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Environment Settings', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input_path', dest = 'input_path', type = str, default = 'F:/ResearchProjects/RIPCODES/Sample_Data/Dehaze/2017-06-28/Dehazed_Img/',
                            help='Path to where files are saved')

    parser.add_argument('-o', '--out_path', dest = 'out_path', type = str, default = 'F:/ResearchProjects/RIPCODES/Sample_Data/Ortho/2017-06-28/',
                            help='Processed Img Path')

    args = parser.parse_args()

    input_dir = args.input_path
    out_dir = args.out_path
    
    dests = ['Dehazed_Img', 'Masks', 'Dark_Channel_Prior', 'Dehazed_Gamma_Img', 'EqualHist']
    
    CheckIfFolderExist(out_dir)
    
    subs = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    for s in subs:
        temp_out_dir = out_dir + s.split('/')[-1] + '/'
        CheckIfFolderExist(temp_out_dir)
        temp_input_dir = s + '/'
        for d in dests:
            CheckIfFolderExist(temp_out_dir + d + '/')
        group_dehaze_processing(temp_input_dir, 'jpg', temp_out_dir, dests)
