# Inpainting Pipeline
The inpainting pipeline utilizes a large number of packages, and therefore needs some installation before it can be used.
## Weights
Various models and weights can be downloaded from this [drive folder.](https://drive.google.com/drive/folders/1x8Nwt0mXi0SEdQMadOrEQXmhrTwPwgan?usp=sharing) 
Download them all and unzip them into the Inpainting folder.
## Environment Setup
The environment can be setup using miniconda. If it is not installed, do the following:

    sudo apt-get update && \
    sudo apt-get upgrade -y && \
    sudo apt-get install unzip libeigen3-dev ffmpeg build-essential nvidia-cuda-toolkit
    
    mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm -rf ~/miniconda3/miniconda.sh && \
    ~/miniconda3/bin/conda init bash && \
    ~/miniconda3/bin/conda init zsh

You can then install the conda environment as such:
```
conda env create -f environment.yml
conda activate inpainting
python3 -m pip install -r requirements.txt
```
## Running the Pipeline
The pipeline can be run with a single file, inpainting.py. First, place input images in /img/test/, and run the following:

    python3 inpainting.py --image-folder img/test/ --output-folder gen_image/

The outputs will be in the folder gen_image, which can then be used for SIFU.
