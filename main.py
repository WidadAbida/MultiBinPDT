import embedding
from embedding import *
import argparse
import cv2
import time 
import numpy as np
import matplotlib.pyplot as plt
import copy


def ArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpath',help='path citra sampul')
    parser.add_argument('--spath', help='path citra rahasia')
    parser.add_argument('--csize', help='ukuran citra sampul')
    parser.add_argument('--ssize', help='ukuran citra rahasia')
    # parser.add_argument()

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = ArgParser()
    cpath = args.cpath
    spath = args.spath
    csize = int(args.csize)
    ssize = int(args.ssize)

    ##### Citra Rahasia
    #secImg = loadSecretIm(f".\secret\{spath}.BMP")
    secImg = loadSecretIm(f"C:/Users/ASUS/Downloads/stegano/Code/secret/{spath}.BMP")
    
    resize = imgResize(secImg, ssize)
    bits_sec_msg = imageToBits(resize)
    cv2.imwrite(f"C:/Users/ASUS/Downloads/stegano/Code/secret/resize/{spath}_{ssize}.png", secImg)

    ##### CITRA SAMPUL
    # Load and Halftone
    img = loadImage(f"C:/Users/ASUS/Downloads/stegano/Code/sampul/{cpath}.tiff", csize)
    ht_img = convertToHalftone(img, csize)

    # Membagi Menjadi Blok Tak Bertumpukan
    blocks4x4 = dividing_blocks(ht_img, 4, 4)
    
    # Menghitung Densitas Piksel Setiap Blok -> Bisa Dibangun Histogram
    pix_dens4x4 = pixel_density(blocks4x4, 4 , 4, csize)

    # Save ht_img
    cv2.imwrite(f"C:/Users/ASUS/Downloads/stegano/Code/halftone/{cpath}_ht_{csize}.png", ht_img)


    ### PDP
    # Mencari PDP (Couple) -> r = 2 dan dimulai dari array ke 1/3 sampai 2/3
    pair = pixel_density_pair(pix_dens4x4, 2,5) 
    
    # Mengurutkan Histogram -> Diurutkan karena saat perhitungan PDP densitas urut kecil ke besar
    sorted_hist = sort_histogram(pix_dens4x4)

    # Pasangan Dua Densitas
    dens_pair = [int(sorted_hist[pair][0]), int(sorted_hist[pair+1][0])]

    # Mencari PDP (Trio) -> r = 2 dan dimulai dari array ke 1/3 sampai 2/3
    pair_trio = pixel_density_pair_multi(pix_dens4x4, 2, 5)

    # Pasangan Tiga Densitas
    dens_pair_trio = [int(sorted_hist[pair_trio][0]), int(sorted_hist[pair_trio+1][0]), int(sorted_hist[pair_trio+2][0])]

    # Blok Pembawa (masih urut)
    carrier_blocks = [i for i, x in enumerate(pix_dens4x4) if x == dens_pair[0] or x == dens_pair[1]]
    carrier_blocks_trio = [i for i, x in enumerate(pix_dens4x4) if x == dens_pair_trio[0] or x == dens_pair_trio[1] or x == dens_pair_trio[2]]

    # List PD -> Digunakan Untuk PD Check Trio -> Daripada hitung PD satu satu lagi
    pd_list = [x for i, x in enumerate(pix_dens4x4) if x == dens_pair_trio[0] or x == dens_pair_trio[1] or x == dens_pair_trio[2]]

    # Index Blok Pembawa Diacak Dengan Seed 
    carrier_blocks_shuf = shuffle_list(np.copy(carrier_blocks), 10)
    carrier_blocks_shuf_trio = shuffle_list(np.copy(carrier_blocks_trio), 10)

    # PDT Diacak Dengan Seed 
    pd_list_shuf = shuffle_list(np.copy(pd_list), 10)

    # Index Blok Pembawa -> Menjadi Index Berformat Row Column
    carrier_indices_shuf = row_col(blocks4x4 , carrier_blocks_shuf)
    carrier_indices_shuf_trio = row_col(blocks4x4 , carrier_blocks_shuf_trio)

    #####  Citra rahasia untuk SDCS 
    # Pesan Rahasia W -> Array 2D -> lm x 4
    msg_W = [bits_sec_msg[i:i+4] for i in range(0,len(bits_sec_msg),4)]

    # Barisan Blok X -> Array 2D -> lb x 4
    block_seq_X = [carrier_blocks_shuf_trio[i:i+4] for i in range(0,len(carrier_blocks_shuf_trio),4)]

    ##### Embedding
    # Pengecekan Kondisi (PDP)
    rule_check = Condition_Check(blocks4x4, bits_sec_msg, carrier_indices_shuf, dens_pair)

    # Pengecekan Kondisi (PDP Trio) 
    # Pertama -> Dicek dengan SDCS -> di metode ini SDCS dipanggil
    rule_check_trio = Condition_Check_Multi(block_seq_X, msg_W) # Array 2D -> l x 4
    rule_check_trio= np.array(rule_check_trio).flatten() # Array 1D

    # Kedua -> Dicek Apakah PD perlu ditambah atau dikurang
    pd_check_trio = PD_Condition_Check(rule_check_trio, pd_list_shuf, dens_pair_trio)


    # Q original image
    TQ_0 = Tsv(ht_img, csize, csize)
    MQ_0 = M_all(TQ_0, csize-2, csize-2)
    Q_0 = QAll(MQ_0)


    # Meng-copy sehingga citra asli tidak berubah
    ht_img_pdt = copy.deepcopy(ht_img)
    b4x4_pdt = dividing_blocks(ht_img_pdt, 4, 4)

    ht_img_sdcs = copy.deepcopy(ht_img)
    b4x4_sdcs = dividing_blocks(ht_img_sdcs, 4, 4)

    #Embedding PDT
    start_pdt = time.time()
    imgY_pdt = pixel_density_transition(ht_img_pdt, b4x4_pdt, carrier_indices_shuf, rule_check, Q_0)
    end_pdt = time.time()

    cv2.imwrite(f"C:/Users/ASUS/Downloads/stegano/Code/stego/{cpath}_{csize}_{ssize}_pdt.png", imgY_pdt)

    #Embedding SDCS
    start_sdcs = time.time()
    imgY_sdcs = sdcs_pixel_density_transition(ht_img_sdcs, b4x4_sdcs, carrier_indices_shuf_trio, pd_check_trio, Q_0)
    end_sdcs = time.time()

    cv2.imwrite(f"C:/Users/ASUS/Downloads/stegano/Code/stego/{cpath}_{csize}_{ssize}_sdcs.png", imgY_sdcs)

    # Menulis txt untuk PDT
    op_pdt = f"C:/Users/ASUS/Downloads/stegano/Code/durasi/{cpath}_{spath}_{csize}_{ssize}_pdt.txt"
    with open(op_pdt, 'a') as f:
        f.write(f"PDP               = {dens_pair}\n")
        f.write(f"Banyak Blok       = {sorted_hist[pair]} dan {sorted_hist[pair+1]}\n")
        f.write(f"Versus citra asli = {(np.sum(np.absolute(np.subtract(ht_img,imgY_pdt)))/255)}\n")
        f.write(f"Durasi PDT        = {(end_pdt-start_pdt)} s\n")

    
    
    #Menulis txt untuk SDCS
    op_sdcs = f"C:/Users/ASUS/Downloads/stegano/Code/durasi/{cpath}_{spath}_{csize}_{ssize}_sdcs.txt"
    with open(op_sdcs, 'a') as f:
        f.write(f"PDP               = {dens_pair_trio}\n")
        f.write(f"Banyak Blok       = {sorted_hist[pair_trio]}, {sorted_hist[pair_trio+1]}, dan {sorted_hist[pair_trio+2]}\n")
        f.write(f"Versus citra asli = {(np.sum(np.absolute(np.subtract(ht_img,imgY_sdcs)))/255)}\n")
        f.write(f"Durasi SDCS       = {(end_sdcs-start_sdcs)} s\n")

