# MultiBinPDT

- Halftone Image Steganography
- Methods : Pixel Density Transition (PDT) and Pixel Density Transition with Sum and Cover Set Differences (SDCS).
- PDT Method is utilizing two bin, while PDT-SDCS Method is utilizing multi bin (three bin is implemented).
- File named `embedding.py` contains all functions needed for embedding process. These functions will be used and called in `main.py` to embed secret messages into cover images.
- File named `script.sh` is bash script to automate embedding process for all cover images to obtain stego images .
- File named `extraction.py` contains all functions needed for extraction process. This functions will be used and called in `main_extract.py` to extract secret message from stego images.
