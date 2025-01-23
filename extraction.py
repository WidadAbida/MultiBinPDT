#Import Library
import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image as im
import copy


def loadStegoImage(path):
    #Load image as a unchanged image
    images = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    #thresholding soalnya ada yang kedetek sebagai  selain sbg 0 dan 255
    ret, thresh = cv2.threshold(images, 127, 255, cv2.THRESH_BINARY)
    
    return thresh

def dividing_blocks(img, m, n):
    h, w = img.shape
    blocks = [[img[i:i+int(m),j:j+int(n) ] for j in range(0, w, n)]for i in range(0, h, m) ]
    
    return blocks

# def rearrange_blocks(arr): # GA DIPAKE
#     new_arr = []
#     l = len(arr)
#     for i in range(l):
#         new_arr.append(np.concatenate((arr[i][0:l]),axis=1))
    
#     new_img = []
#     for i in range(len(new_arr)):
#         for j in range(len(new_arr[i])):
#             new_img.append(np.array(new_arr[i][j]))
#     return np.array(new_img)

def pixel_density(block, m, n, size): # m x n ukuran blok
    dens= []
    for i in range(int(size/m)): # antara pake m n or pake len
        result = 0
        for j in range(int(size/n)):
            result = sum(block[i][j].flatten())/255
            dens.append(m*n-result)
    
    return dens

def shuffle_list(arr, k):
    random.seed(k)
    random.shuffle(arr)
    return arr


def one_block_pd(block):
    m, n = block.shape
    result = sum(block.flatten())/255
    return m*n -result

def sort_histogram(arr):
    sorted = np.array (pd.Series(arr).value_counts().reset_index().values.tolist())
    sorted = sorted[sorted[:,0].argsort()]
    return sorted

def row_col(blok, arr_index):
    l = len(blok)
    result = []
    for i in range(len(arr_index)):
        div = int(arr_index[i]/l)
        mod = arr_index[i] % l
        result.append([div, mod])
    return result


# Fungsi untuk menampilkan histogram dari pixel densitas
def PDH_show(density):
    hist= pd.Series(density).value_counts().reset_index().values.tolist()
    x = [i[0] for i in hist]
    y = [i[1] for i in hist]

    #print(hist)
    #print(x, len(x))
    #print(y, len(y))
    plt.bar(x,y)
    plt.show()


def pixel_density_pair(arr, r, part):
    sorted = np.array (pd.Series(arr).value_counts().reset_index().values.tolist())
    sorted = sorted[sorted[:,0].argsort()]
    
    # k = len(sorted)
    k = int(len(sorted)/part)

    # | (f(k) - f(k+1) |
    # num = np.absolute(np.subtract(sorted[0:k-1,1], sorted[1:k,1]))
    # num = np.absolute([ sorted[i][1] - sorted[i+1][1] for i in range(k-1) ])
    num = np.absolute(np.subtract(sorted[k:k*(part-1)-1,1], sorted[k+1:k*(part-1),1]))
    
    # ( (f(k) + f(k+1) )^r
    # denom = np.power(np.add(sorted[0:k-1,1], sorted[1:k,1]), r)
    denom = np.power(np.add(sorted[k:k*(part-1)-1,1], sorted[k+1:k*(part-1),1]), r)

    # argmin
    result = np.argmin(np.divide(num,denom))

    # return result
    return k+result #lokasi

def pixel_density_pair_multi(arr, r, part):
    sorted = np.array (pd.Series(arr).value_counts().reset_index().values.tolist())
    sorted = sorted[sorted[:,0].argsort()]
    #print(sorted)

    # k = len(sorted)
    k = int(len(sorted)/part)

    # | (f(k) - f(k+1) | + | (f(k+1) - f(k+2) | + | (f(k) - f(k+2) |
    num  = np.absolute(np.subtract(sorted[k:k*(part-1)-2,1],   sorted[k+1:k*(part-1)-1,1]))
    num2 = np.absolute(np.subtract(sorted[k+1:k*(part-1)-1,1], sorted[k+2:k*(part-1),1]))
    num3 = np.absolute(np.subtract(sorted[k:k*(part-1)-2,1],   sorted[k+2:k*(part-1),1]))

    # ( (f(k) + f(k+1) +f(k+2) )^r
    denom = np.power(np.add(np.add(sorted[k:k*(part-1)-2,1], sorted[k+1:k*(part-1)-1,1]), sorted[k+2:k*(part-1),1]), r)
    
    # argmin
    result = np.argmin(np.divide(np.add(np.add(num, num2), num3),denom))

    # return result
    return k+result #lokasi




# PDT ======================
def extraction_process(blocks, index, PDP): # PDT
    bits = []
    for i in range(len(index)):
        div = index[i][0]
        rem = index[i][1]
        if one_block_pd(blocks[div][rem]) == PDP[0]:
            bits.append(0)
        elif one_block_pd(blocks[div][rem]) == PDP[1]:
            bits.append(1)
    return bits

# PDT ======================

# SDCS ======================
def DecimalToBinaryArray(dArr, msize): 
    binary = [format(n, '04b') for n in dArr]
    res = []
    for i in range(len(dArr)):
        for j in binary[i]:
            res.append(int(j))
    return res


def sdcs_extraction_process(blocks, index, pdens, PDP, msize): #SDCS
    res = []
    # Bagi jadi 4 blok
    carrier = copy.deepcopy(index)
    for i in range(msize):
        if pdens[i] == PDP[0]:
            carrier[i] = carrier[i] - 1
        elif pdens[i] == PDP[1]:
            continue
        elif pdens[i] == PDP[2]:
            carrier[i] = carrier[i] + 1
        else:
            print('Invalid PD')

    block_Y = [carrier[i:i+4] for i in range(0,msize,4)]

    # Extraksi dengan SDCS
    A = [1, 3, 9, 11]
    res = [np.sum(np.multiply(A, item))%16 for item in block_Y ]
    
    # Check Densitas
    return res
# SDCS ======================

def rearrange_sec_img(arr, m, n):
    return np.reshape(arr, (m,n))