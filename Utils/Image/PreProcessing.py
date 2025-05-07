# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:27:30 2022

@author: bbean
"""

import cv2
import os
import argparse
import scipy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform

def CheckIfDirExist(path):
    if os.path.exists(f'{path}'):
        pass
    else:
        os.mkdir(f'{path}')

def LoadMatLabFile(mpath):
    return scipy.io.loadmat(mpath)

def ApplyFourPointTransformation(input_folder, input_name, four_points, out_folder, out_name):
    img = cv2.imread(input_folder + input_name)
    rect = four_point_transform(img, np.array(four_points))
    CheckIfDirExist(out_folder)
    cv2.imwrite(out_folder + out_name, rect)

def InterPolation(orig_img, orig_x_min, orig_x_max, inter_x, inter_y):
    pass
        
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


def Defog(m, r, eps, w, maxV1):                 # 输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, 2)                           # 得到暗通道图像
    Dark_Channel = zmMinFilterGray(V1, 7)
    # cv2.imwrite('3.png', Dark_Channel * 255)
    V1 = guidedfilter(V1, Dark_Channel, r, eps)  # 使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)                  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
    V1 = np.minimum(V1 * w, maxV1)               # 对值范围进行限制
    return V1, A


def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    Mask_img, A = Defog(m, r, eps, w, maxV1)             # 得到遮罩图像和大气光照
    # cv2.imwrite('2.png', Mask_img * 255)
    for k in range(3):
        Y[:,:,k] = (m[:,:,k] - Mask_img)/(1-Mask_img/A)  # 颜色校正
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))       # gamma校正,默认不进行该操作
    return Y

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

def group_processing(input_dir, output_dir, des):
    suffix = 'jpg'
    CheckIfDirExist(output_dir + des + '_01/')
    CheckIfDirExist(output_dir + des + '_02/')
    CheckIfDirExist(output_dir + des + '_03/')
    CheckIfDirExist(output_dir + des + '_04/')
    CheckIfDirExist(output_dir + des + '_05/')
    CheckIfDirExist(output_dir + des + '_06/')
    CheckIfDirExist(output_dir + des + '_07/')
    CheckIfDirExist(output_dir + des + '_08/')

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            filepath = os.path.join(root, file)
            # Here, split the path to get the suffix of file
            filesuffix = os.path.splitext(filepath)[1][1:]
            # If suffix matches, read the image.
            if filesuffix in suffix:
                img = cv2.imread(filepath)
                hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                Y,M,D = deHaze(img / 255)
                
                Y1, M1, D1 = deHaze(hsv_img / 255) 
                cv2.imwrite(output_dir + '/' + des + '_01/' + file, Y * 255)
                cv2.imwrite(output_dir + '/' + des + '_02/' + file, M * 255)
                cv2.imwrite(output_dir + '/' + des + '_03/' + file, D * 255)
                cv2.imwrite(output_dir + '/' + des + '_04/' + file, hsv_img)
                
                Y = ((Y+1)*255).astype('uint8')
                cv2.imwrite(output_dir + '/' + des + '_05/' + file, cv2.cvtColor(Y, cv2.COLOR_BGR2HSV))
                cv2.imwrite(output_dir + '/' + des + '_06/' + file, Y1 * 255)
                cv2.imwrite(output_dir + '/' + des + '_07/' + file, M1 * 255)
                cv2.imwrite(output_dir + '/' + des + '_08/' + file, D1 * 255)
                
def image_division(indir, suffix, h1, h2, w1, w2, out_dir):
    # indir: direction to your folder.
    # suffix: suffix of your image, e.g .png, .jpg.
    # h1,h2: range that you want to crop for the height.(0~1)
    # w1,w2: range that you want to crop for the width.(0~1)
    for root, dirs, files in os.walk(indir):
        for file in files:
            filepath = os.path.join(root, file)
            filesuffix = os.path.splitext(filepath)[1][1:]
            if filesuffix in suffix:
                image = cv2.imread(filepath)
                size = image.shape
                height,width = size[:2]
                h1_len = max(round(h1*height),1)
                h2_len = min(round(h2*height),height)
                w1_len = max(round(w1*width),1)
                w2_len = min(round(w2*width),width)
                cropped = image[h1_len:h2_len,w1_len:w2_len]
                cv2.imwrite(out_dir + '/'+ file, cropped)

def getAllFilesOfASuffixWithinAFolder(input_dir, suffix):
    res = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            filesuffix = os.path.splitext(input_dir)[1][1:]
            if filesuffix == suffix:
                res.append(file)
    return res

def getAllSubDirectoryInADir(mother_folder):
    dbtype_list = os.listdir(mother_folder)
    for dbtype in dbtype_list:
        if os.path.isfile(os.path.join(mother_folder, dbtype)):
            dbtype_list.remove(dbtype)
    return dbtype_list

def CreateClassificationFileBasedOnFolders(mother_folder, subfolder_list, file_suffix):
    res = {}
    ref = {}
    i = 0
    for sf in subfolder_list:
        ref[sf] = i
        folder = mother_folder + sf
        file_list = getAllFilesOfASuffixWithinAFolder(folder, file_suffix)
        for f in file_list:
            res[f] = sf
        i = i + 1
    return res, ref

def SaveClassificationRes(res_dict, ref_dict, save_folder1, save_file1, save_folder2, save_file2):
    CheckIfDirExist(save_folder1)
    CheckIfDirExist(save_folder2)
    with open(save_folder1 + save_file1, 'wb') as f:
        for k in res_dict.keys():
            temp_key = str(k)
            temp_classifier = res_dict.get(k)
            print('%s %s\n'%(temp_key, temp_classifier), file = f)
        f.close()
    with open(save_folder2 + save_file2, 'wb') as g:
        for k in res_dict.keys():
            temp_key = str(k)
            temp_classifier = res_dict.get(k)
            temp_ind = ref_dict.get(temp_classifier)
            print('%s %d\n'%(temp_key, temp_ind), file = g)
        g.close()

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Environment Settings', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-ip', '--input_path', dest = 'input_path', type = str, \
                        default = './', help='Path to original imgs')
    
    parser.add_argument('-cp', '--cut_path', dest = 'cut_path', type = str, \
                        default = './Cut/', help='Path to save cut imgs')
    
    parser.add_argument('-de', '--des', dest = 'des', type = str, default = 'des', help = 'Prefix of destination folder')
        
    parser.add_argument('-dp', '--defog_path', dest = 'defog_path', type = str, \
                        default = './Defog/', help='Path to save defogged imgs')
    
    parser.add_argument('-op', '--ortho_path', dest = 'ortho_path', type = str, \
                        default = './Ortho/', help='Defogging Destination')
    
    args = parser.parse_args()
    