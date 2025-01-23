import extraction
from extraction import *
import argparse
import cv2
import time 
import numpy as np
import matplotlib.pyplot as plt
import copy

def ArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ppath',help='path stego pdt')
    parser.add_argument('--sdpath', help='path stego sdcs')
    parser.add_argument('--stsize', help='ukuran citra stego')
    parser.add_argument('--msize', help='ukuran citra yg disembunyikan')
    # parser.add_argument()

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = ArgParser()
    pdt_path = args.ppath
    sdcs_path = args.sdpath
    stsize = int(args.stsize)
    msize = int(args.msize)
    

    stego_img_pdt = loadStegoImage(f"C:/Users/ASUS/Downloads/stegano/Code/stego/{pdt_path}_{stsize}_{msize}_pdt.png")
    stego_img_sdcs = loadStegoImage(f"C:/Users/ASUS/Downloads/stegano/Code/stego/{sdcs_path}_{stsize}_{msize}_sdcs.png")
    # stego_img_pdt = loadStegoImage(f"/home/widadar/stego2/{pdt_path}_{stsize}_{msize}_pdt.png")
    # stego_img_sdcs = loadStegoImage(f"/home/widadar/stego2/{sdcs_path}_{stsize}_{msize}_sdcs.png")
    op_pdt = f"/home/widadar/stego2/{pdt_path}_{stsize}_{msize}_pdt.png"
    op_sdcs = f"/home/widadar/stego2/{sdcs_path}_{stsize}_{msize}_pdt.png"
    # Bagi blok
    b4x4_pdt = dividing_blocks(stego_img_pdt, 4, 4)
    b4x4_sdcs = dividing_blocks(stego_img_sdcs, 4, 4)

    img_size =len(stego_img_sdcs)
    # Hitung histogram
    pdens4x4_pdt = pixel_density(b4x4_pdt, 4 , 4, img_size)
    pdens4x4_sdcs = pixel_density(b4x4_sdcs, 4 , 4, img_size)

    # Mencari PDP -> dimulai dari array ke 1/3 sampai 2/3
    pair = pixel_density_pair(pdens4x4_pdt, 2,5) 
    pair_trio = pixel_density_pair_multi(pdens4x4_sdcs, 2,5) 
    

    # sort the histogram 
    sorted_hist1 = sort_histogram(pdens4x4_pdt)
    sorted_hist2 = sort_histogram(pdens4x4_sdcs)

    # PDP
    dens_pair = [int(sorted_hist1[pair][0]), int(sorted_hist1[pair+1][0])]
    # dens_pair = [9,10 ]

    # PDP trio
    dens_pair_trio = [int(sorted_hist2[pair_trio][0]), int(sorted_hist2[pair_trio+1][0]), int(sorted_hist2[pair_trio+2][0])]
    # dens_pair = [9,10 ]

    with open(op_pdt, 'a') as f:
        f.write()

    # Blok Pembawa (masih urut)
    carrier_blocks = [i for i, x in enumerate(pdens4x4_pdt) if x == dens_pair[0] or x == dens_pair[1]]
    carrier_blocks_trio = [i for i, x in enumerate(pdens4x4_sdcs) if x == dens_pair_trio[0] or x == dens_pair_trio[1] or x == dens_pair_trio[2]]
    pd_list = [x for i, x in enumerate(pdens4x4_sdcs) if x == dens_pair_trio[0] or x == dens_pair_trio[1] or x == dens_pair_trio[2]]

    # Diacak Seed
    carrier_blocks_shuf = shuffle_list(np.copy(carrier_blocks), 10)
    carrier_blocks_shuf_trio = shuffle_list(np.copy(carrier_blocks_trio), 10) 
    pd_list_shuf = shuffle_list(np.copy(pd_list), 10)

    # Index Blok Pembawa -> Menjadi Index Berformat Row Column
    carrier_indices_shuf = row_col(b4x4_pdt , carrier_blocks_shuf)
    carrier_indices_shuf_trio = row_col(b4x4_sdcs , carrier_blocks_shuf_trio)

    # Ekstraksi PDT
    start_pdt = time.time()
    bits_secmsg_pdt = extraction_process(b4x4_pdt, carrier_indices_shuf[:msize*msize], dens_pair)
    bits_secmsg_pdt = np.array(bits_secmsg_pdt)
    start_pdt = time.time()

    #print(bits_secmsg_pdt)
    secret_pixels_pdt = [255*i for i in bits_secmsg_pdt]
    #print(secret_pixels_pdt)


    # Ektraksi SDCS
    start_sdcs = time.time()
    deci_secmsg = sdcs_extraction_process(b4x4_sdcs, carrier_blocks_shuf_trio, pd_list_shuf, dens_pair_trio, msize*msize)
    bits_secmsg_sdcs = DecimalToBinaryArray(deci_secmsg, msize*msize)
    end_sdcs = time.time()
    secret_pixels_sdcs = [255*i for i in bits_secmsg_sdcs]


    secret_img_pdt = rearrange_sec_img(np.array(secret_pixels_pdt), msize, msize)
    secret_img_sdcs = rearrange_sec_img(np.array(secret_pixels_sdcs), msize,msize)

    #op_sdcs = f"C:/Users/ASUS/Downloads/stegano/Code/durasi/{sdcs_path}_{stsize}_{msize}_sdcs.txt"
    with open(op_sdcs, 'a') as f:
        f.write(f"PDP               = {dens_pair_trio}\n")
        f.write(f"Durasi SDCS       = {(end_sdcs-start_sdcs)} s\n")