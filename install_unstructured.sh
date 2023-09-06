#! /bin/bash
pip install unstructured
pip install "unstructured[pdf]"

sudo apt-get update
sudo apt -y install software-properties-common
sudo apt-get update
sudo add-apt-repository -y ppa:alex-p/tesseract-ocr-devel
sudo apt -y install python3-dev
sudo apt -y install python3-opencv
sudo apt -y install tesseract-ocr
sudo apt -y install tesseract-ocr-fra
sudo apt -y install poppler-utils
sudo apt -y install libmagic-dev

# sudo apt -y install swig
# sudo apt -y install ghostscript python3-tk
# pip install "unstructured.PaddleOCR"
