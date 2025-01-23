#Import Library
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import pandas as pd
import copy


#===================KONVERSI===================#
def loadImage(path, size):
    #Load image as a grayscale image
    images = cv2.imread(path, 0)
    
    #convert from BGR to RGB
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    
    #resize image
    images = cv2.resize(images, dsize= (size,size), interpolation=cv2.INTER_CUBIC)

    return images

def convertToHalftone(images, size):
    """return halftone conversion of Grayscale image"""
    """Using halftoning method Error Diffusion Floyd Kernel"""
    # Floyd-Steinberg Kernel
    floyd = [[0/16, 0/16, 7.0/16],
            [3.0/16, 5.0/16, 1.0/16]]
    floyd = np.float32(floyd)
   
    
    # Threshold
    T = 128

    # Slicing (baca lagi) biar bisa dimasukan ke (514,514,3)
    images = images[:, :, 1]

    # Matriks 514 x 514 ukuran kosong buat operation In Out
    # ip = np.zeros((514,514))
    # op = np.zeros((514,514))

    ip = np.zeros((size+2, size+2))
    op = np.zeros((size+2, size+2))

    # Citra diinput ke matriks input
    # ip[1:513, 1:513] = images 
    ip[1:size+1, 1:size+1] = images 

    # Proses Error Diffusion
    for x in range(1, size, 1):
        for y in range(1, size, 1):
            xc = float(ip[x, y])
            xm = float(ip[x, y])
            xc = xc > T
            t = int(xc) * 255
            op[x, y] = t
            err = np.subtract(t, xm)
            f = np.multiply(floyd, err)
            ip[x:x+2, y:y+3] = ip[x:x+2, y:y+3]-f
    
    op = op[1:size+1, 1:size+1]
    return op

def dividing_blocks(img, m, n):
    h, w = img.shape
    blocks = [[img[i:i+int(m),j:j+int(n) ] for j in range(0, w, n)]for i in range(0, h, m) ]
    
    return blocks

def rearrange_blocks(arr):
    new_arr = []
    l = len(arr)
    for i in range(l):
        new_arr.append(np.concatenate((arr[i][0:l]),axis=1))
    
    new_img = []
    for i in range(len(new_arr)):
        for j in range(len(new_arr[i])):
            new_img.append(np.array(new_arr[i][j]))
    return np.array(new_img)

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

# https://crypto.stackexchange.com/questions/78309/how-to-get-the-original-list-back-given-a-shuffled-list-and-seed
def unshuffle_list(arr, k):
    n = len(arr)
    
    # Perm is [1, 2, ..., n]
    perm = [i for i in range(1, n + 1)]

    # Apply sigma to perm
    shuffled_perm = shuffle_list(perm, k)

    # Zip and unshuffle
    zipped_ls = list(zip(arr, shuffled_perm))
    zipped_ls.sort(key=lambda x: x[1])
    
    return [a for (a, b) in zipped_ls]

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


#===================DISPLAY VISUAL===================#
# # Fungsi untuk menampilkan histogram dari pixel densitas
def PDH_show(density):
    hist= pd.Series(density).value_counts().reset_index().values.tolist()
    x = [i[0] for i in hist]
    y = [i[1] for i in hist]

    #print(hist)
    #print(x, len(x))
    #print(y, len(y))
    plt.bar(x,y)
    plt.show()


#===================PDP===================#
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


#===================PMMTM===================#
def state_value(arr, index):
    state_index = np.array([[[0, 1, 8], 
                [0, 2, 16],
                [0, 4, 32]],

                [[32, 4, 0], 
                 [16, 2, 0],
                 [8, 1, 0]],

                [[8,   16, 32],
                 [1,   2, 4],
                 [0, 0, 0]],

                [[0, 0, 0],
                [4,    2,      1],
                [32,    16,      8]],

                [[0, 8,  16],
                [1,    2,  32],
                [0, 4, 0]],

                [[0, 1, 0],
                [4,    2,  8],
                [0, 32,   16]],

                [[16,    32, 0],
                [8,    2,  4],
                [0, 1, 0]],

                [[0, 4, 0],
                [32,     2,     1],
                [16,    8, 0]]])
    temp_arr = copy.deepcopy(arr)
    mul = np.multiply(state_index[index], temp_arr/255)
    sv = np.sum(mul)
    
    return sv

def Tsv(arr, l, w):
    svs = []
    for i in range(l-2): 
        res =[]
        for j in range(w-2): # (HALF SOLVED) Masih belum dikurang satu apakah Tij sudah terhitung blok atau masih pixel
            # print(i, ':', i+3, ' ', j, ':', j+3)
            res.append(state_value(arr[i:i+3, j:j+3], 0))
        svs.append(res)
    return np.array(svs)

def M_all(T, l1, l2):
    M = np.zeros((8, 64,64))

    # Right -> 
    for i in range(l1): # r: 0 - 5 
        for j in range(l2-1): # c: 0 - 5
            r = T[i][j] # -> base
            c = T[i][j+1]
            M[0][int(r)][int(c)] = M[0][int(r)][int(c)] + 1 
    
    # Left -> 
    for i in range(l1): # r: 0 - 5 
        for j in range(l2-1): # c: 0 - 5
            r = T[i][j+1] # -> base
            c = T[i][j]
            M[1][int(r)][int(c)] = M[1][int(r)][int(c)] + 1   

    # Top -> 
    for i in range(l1-1): # ini yang dikurangi satu krn atas bawah
        for j in range(l2): 
            r = T[i+1][j] # -> base
            c = T[i][j]
            M[2][int(r)][int(c)] = M[2][int(r)][int(c)] + 1    

    # Bottom -> 
    for i in range(l1-1): # ini yang dikurangi satu krn atas bawah
        for j in range(l2): 
            r = T[i][j] # -> base
            c = T[i+1][j]
            M[3][int(r)][int(c)] = M[3][int(r)][int(c)] + 1    

    # Right Top -> 
    for i in range(l1-1): # ini yang dikurangi satu krn atas
        for j in range(l2-1): # ini dikurangi satu krn kanan
            r = T[i+1][j] # -> base
            c = T[i][j+1]
            M[4][int(r)][int(c)] = M[4][int(r)][int(c)] + 1    

    # Right Bottom -> 
    for i in range(l1-1): # ini yang dikurangi satu krn atas
        for j in range(l2-1): # ini dikurangi satu krn kanan
            r = T[i][j] # -> base
            c = T[i+1][j+1]
            M[5][int(r)][int(c)] = M[5][int(r)][int(c)] + 1    

    # Left Top -> 
    for i in range(l1-1): # ini yang dikurangi satu krn atas
        for j in range(l2-1): # ini dikurangi satu krn kanan
            r = T[i+1][j+1] # -> base
            c = T[i][j]
            M[6][int(r)][int(c)] = M[6][int(r)][int(c)] + 1    

    # Left Bottom -> 
    for i in range(l1-1): # ini yang dikurangi satu krn atas
        for j in range(l2-1): # ini dikurangi satu krn kanan
            r= T[i][j+1] # -> base
            c = T[i+1][j]
            M[7][int(r)][int(c)] = M[7][int(r)][int(c)] + 1    
    
    #### Transition
    Mtrans = np.zeros((8,64,64))
    for i in range(8):
        for j in range(64):
            if (np.sum(M[i][j]) == 0):
                continue
            Mtrans[i][j]= np.divide(M[i][j], np.sum(M[i][j]))
    
    return Mtrans

def QAll(M):
    res = 1/8 * (M[0]+M[1]+M[2]+M[3]+M[4]+M[5]+M[6]+M[7])
    return res

#### PDT
def one_block_pd(block):
    m, n = block.shape
    result = sum(block.flatten())/255
    return m*n -result

def Condition_Check(block, bmessage, loc, dpair):
    l = len(block)
    lb = len(bmessage)
    result = []
    # print(l, len(loc))
    for i in range(len(loc)): #-> sementara pakai punya shuffle block index
        if i >= lb:
            break
        # div = int(loc[i]/l)
        # rem = loc[i]%l
        div = loc[i][0]
        rem = loc[i][1]
        #print(div, rem, i)
        if one_block_pd(block[div][rem]) == dpair[0] and bmessage[i] == 1:
            result.append(1)
            #print("lower and 1", one_block_pd(block[div][rem]), bmessage[i])
        elif one_block_pd(block[div][rem]) == dpair[1] and bmessage[i] == 0:
            result.append(-1)
            #print("higher and 0", one_block_pd(block[div][rem]), bmessage[i])

        else:
            result.append(0)
            #print("tetap", one_block_pd(block[div][rem]),bmessage[i])
    return result # mengembalikan apakah dia perlu dibalik pikselnya dan piksel yang mana

### PDT ###
def pixel_marking(block):
    hmark = []
    pmark = []
    for i in range(len(block)):
        for j in range(len(block[i])):
            index = []
            index.append(i)
            index.append(j)
            if block[i][j] == 0: #hitam
                hmark.append(index)
            else:
                #continue
                pmark.append(index)


    return hmark, pmark

# Hitam 0: putih:1
# HitamToPutih -> 1 to 0 -> operasi -1 -> memakai hmark
def Delta_HtoP(img, arr, index, mark, Q0): #  shuf_index, marked index
    delta= []
    l = len(mark)
    Qnew = []
    size = len(img)
    #temp_reg = copy.deepcopy(arr)
    temp_reg = img
    row = index[0] * len(arr[0][0])
    col = index[1] * len(arr[0][0])
    for i in range(l):
        if temp_reg[row+mark[i][0]][col+mark[i][1]] != 0.0:
            print('Salah Warna  Pixel')
        temp_reg[row+mark[i][0]][col+mark[i][1]] = 255.0 # 0 to 1 -> h to p
        Ti = Tsv(temp_reg, size, size)
        Mi = M_all(Ti, size-2, size-2)
        Qi = QAll(Mi)
        diff = np.subtract(Qi, Q0)
        delta.append(np.sum(diff))
        Qnew.append(Qi)
        temp_reg[row+mark[i][0]][col+mark[i][1]] = 0.0 # dibalikin nilainya
    optimal = np.argmin(delta)
    return Qnew[optimal], optimal

# PutihToHitam -> 0 to 1 -> operasi +1 -> memakai pmark
def Delta_PtoH(img, arr, index, mark, Q0): # marked index
    delta= []
    l = len(mark)
    Qnew = []
    size = len(img)
    #temp_reg = copy.deepcopy(arr)
    temp_reg = img
    row = index[0] * len(arr[0][0])
    col = index[1] * len(arr[0][0])
    for i in range(l):
        if temp_reg[row+mark[i][0]][col+mark[i][1]] != 255.0:
            print('Salah Warna  Pixel')
        temp_reg[row+mark[i][0]][col+mark[i][1]] = 0.0
        Ti = Tsv(temp_reg, size, size)
        Mi = M_all(Ti, size-2, size-2)
        Qi = QAll(Mi)
        diff = np.subtract(Qi, Q0)
        delta.append(np.sum(diff))
        Qnew.append(Qi)
        temp_reg[row+mark[i][0]][col+mark[i][1]] = 255.0
    optimal = np.argmin(delta)
    return Qnew[optimal], optimal

# ht img, shuf_index, rule-check, Q0
def pixel_density_transition(image, blocks, index, cond, Q0):
    # sepertinya pakai do-while loop saja
    l = len(blocks[0][0])
    Qcurrent = Q0
    for i in range(len(index)):
        if i >= len(cond):
            break
        div = index[i][0]
        rem = index[i][1]
        hmark, pmark = pixel_marking(blocks[div][rem])

        if cond[i] == 0:
            #print('0')
            continue
        elif cond[i] == 1:
            #print('1')
            newQ, best = Delta_PtoH(image, blocks, index[i], pmark, Qcurrent)
            #print(best)
            #blocks = swapping_pixel(blocks, index[i], pmark[best], cond)
            #print('1', pmark[best], image[(div*l)+pmark[best][0]][(rem*l)+pmark[best][1]] )
            image[div*l+pmark[best][0]][(rem*l)+pmark[best][1]] = 0.0
            Qcurrent = newQ
        elif cond[i] == -1:
            #print('-1')
            newQ, best = Delta_HtoP(image, blocks, index[i], hmark, Qcurrent)
            #print(best)
            #blocks = swapping_pixel(blocks, index[i], hmark[best], cond)
            #print('-1', hmark[best], image[(div*l)+hmark[best][0]][(rem*l)+hmark[best][1]] )
            image[(div*l)+hmark[best][0]][(rem*l)+hmark[best][1]] = 255.0
            Qcurrent = newQ
        else:
            return False
    return image
    
#===================PENYEMBUNYIAN SDCS===================#

def binaryToDecimal(seq):
    res = ''.join([str(item) for item in seq])
    return int(res,2)

def iterasi():
    #brute force for real sampai dp bisa buat negatif
    iter = []
    for i in range(3):
        for j in range(3):
            for k in range (3):
                for l in range(3):
                    iter.append([i-1,j-1,k-1,l-1])   
    return iter


def SDCS(X, W):
    A = [1, 3, 9, 11]
    A_X = np.sum(np.multiply(A, X)) % 16
    W_AX = binaryToDecimal(W) -  A_X
    S = []

    # 1, 3, 9, 11, -1, -3, -9, -11
    # ada himpunan kosong () dan semuanya (1, 3, 9, 11, -1, -3, -9, -11)
    # cari himpunan bagian yang kalau ditambah jadi 
    iter = iterasi()

    # Mencari S -> A*S = W_AX
    i = 0
    while i < len(iter):
        res = np.sum(np.multiply(iter[i],A))
        if res == W_AX:
            S = iter[i]
            break
        i+=1
    return S


# Batasnya -> panjang pesan -> KALAU SEMISAL BLOKNYA KURANG ???
def Condition_Check_Multi(block_X, msg):
    res = []
    # Batasnya -> panjang pesan -> KALAU SEMISAL BLOKNYA KURANG
    for i in range(len(msg)):
        sdcs_check = SDCS(block_X[i], msg[i])
        res.append(sdcs_check)
    return res

def PD_Condition_Check(cond, pd_seq, dpair):
    res = []
    for i in range(len(cond)):
        # Jika Kondisi SDCS -> 0 
        if cond[i] == 0:
            if pd_seq[i] == dpair[0]: #  PD low
                res.append(1)
            elif pd_seq[i] == dpair[1]: # PD medium
                res.append(0)
            elif pd_seq[i] == dpair[2]:
                res.append(-1)
            else:
                print("Invalid PD Seq")
        elif cond[i] == 1:
            if pd_seq[i] == dpair[0]: #  PD low
                res.append(2)
            elif pd_seq[i] == dpair[1]: # PD medium
                res.append(1)
            elif pd_seq[i] == dpair[2]:
                res.append(0)
            else:
                print("Invalid PD Seq")
        elif cond[i] == -1:
        
            if pd_seq[i] == dpair[0]: #  PD low
                res.append(0)
            elif pd_seq[i] == dpair[1]: # PD medium
                res.append(-1)
            elif pd_seq[i] == dpair[2]:
                res.append(-2)
            else:
                print("Invalid PD Seq")
        else:
            print("Invalid SDCS Condition")
    return res

def sdcs_pixel_density_transition(image, blocks, index, condPD, Q0):
    size = len(image)
    # sepertinya pakai do-while loop saja
    l = len(blocks[0][0])
    Qcurrent = Q0
    for i in range(len(index)):
        if i >= len(condPD):
            break
        div = index[i][0]
        rem = index[i][1]
        hmark, pmark = pixel_marking(blocks[div][rem])
        if condPD[i] == 0:
                #print('PD:tetap')
                continue
        elif condPD[i] == 1: # +1 -> dari putih ke hitam
                #best = Delta_PtoH(image, blocks, index[i], pmark, Q0, size)
                newQ, best = Delta_PtoH(image, blocks, index[i], pmark, Qcurrent)
                #print('PD:+1', pmark[best], image[(div*l)+pmark[best][0]][(rem*l)+pmark[best][1]] )
                image[div*l+pmark[best][0]][(rem*l)+pmark[best][1]] = 0.0
                Qcurrent=newQ
        elif condPD[i] == -1: # -1 -> dari hitam ke putih
                newQ, best = Delta_HtoP(image, blocks, index[i], hmark, Qcurrent)
                #print('PD:-1', hmark[best], image[(div*l)+hmark[best][0]][(rem*l)+hmark[best][1]] )
                image[(div*l)+hmark[best][0]][(rem*l)+hmark[best][1]] = 255.0
                Qcurrent=newQ

        elif condPD[i] == 2: # +2 -> dari putih ke hitam x 2
                newQ, best = Delta_PtoH(image, blocks, index[i], pmark, Qcurrent)
                #print('PD:+2(1)', pmark[best], image[(div*l)+pmark[best][0]][(rem*l)+pmark[best][1]] )
                image[div*l+pmark[best][0]][(rem*l)+pmark[best][1]] = 0.0
                Qcurrent=newQ

                # pmark prlu dipop
                pmark.pop(best)

                newQ, best = Delta_PtoH(image, blocks, index[i], pmark, Qcurrent)
                #print('PD:+2(1)', pmark[best], image[(div*l)+pmark[best][0]][(rem*l)+pmark[best][1]] )
                image[div*l+pmark[best][0]][(rem*l)+pmark[best][1]] = 0.0
                Qcurrent=newQ
        elif condPD[i] == -2: # -1 -> dari hitam ke putih x 2
                newQ, best = Delta_HtoP(image, blocks, index[i], hmark, Qcurrent)
                #print('PD:-2(1)', hmark[best], image[(div*l)+hmark[best][0]][(rem*l)+hmark[best][1]] )
                image[(div*l)+hmark[best][0]][(rem*l)+hmark[best][1]] = 255.0
                Qcurrent=newQ

                # hmark sudah berubah satu -> perlu di pop sebelum digunakan lagi
                hmark.pop(best)

                newQ, best = Delta_HtoP(image, blocks, index[i], hmark, Qcurrent)
                #print('PD:-2(2)', hmark[best], image[(div*l)+hmark[best][0]][(rem*l)+hmark[best][1]] )
                image[(div*l)+hmark[best][0]][(rem*l)+hmark[best][1]] = 255.0
                Qcurrent=newQ
        else:
                print('Invalid Condition')
    return image


#===================CITRA RAHASIA===================#

def loadSecretIm(path):
    #Load image as a grayscale image
    images = cv2.imread(path, 0)
    
    #convert from BGR to RGB
    #images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

    ret, thresh = cv2.threshold(images, 127, 255, cv2.THRESH_BINARY)
    
    return thresh

def loadSecretImRz(path, size):
    #Load image as a grayscale image
    images = cv2.imread(path, 0)
    
    #convert from BGR to RGB
    #images = cv2.cvtColor(images, cv2.COLOR_GRAY2GRAY)
    
    images = cv2.resize(images, dsize= (size,size), interpolation=cv2.INTER_CUBIC)
    ret, thresh = cv2.threshold(images, 127, 255, cv2.THRESH_BINARY)
    return thresh

def imgResize(images, size):
    images = cv2.resize(images, dsize= (size,size), interpolation=cv2.INTER_CUBIC)
    ret, thresh = cv2.threshold(images, 127, 255, cv2.THRESH_BINARY)
    return thresh
# Karena binary  image jadi tidak perlu diconvert. langsung 255 jadi bit 1 dan 0 tetap 0
def imageToBits(img):
    l, w = img.shape
    res = []
    for i in range(l):
        for j in range(w):
            if int(img[i][j]) == 255:
                res.append(1)
            elif int(img[i][j]) == 0:
                res.append(0)
            else:
                print("salah tipe citra")
    return res

# REARRANGE SECRET MESSAGE

def rearrange_sec_bits(ext_bits):
    str_bits = [str(item) for item in ext_bits]
    print(len(str_bits), str_bits, type(str_bits))
    
    # agar satu huruf/angka penuh jadi dibagi 8
    l = int(len(ext_bits)/8)
    msg = []
    for i in range(l):
        ind = str_bits[i*8: (i+1)*8]
        msg.append(''.join(ind))

    # Converr from binary string to int
    arr_msg = [int(item, 2) for item in msg]

    return arr_msg

def rearrange_sec_img(arr, size):
    return np.reshape(arr, (size,size))