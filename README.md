## WP3: Deep Learning Model For LandSlide Detection using Sentinel-2 Remote Sensing Image

## General:

1/ To run this model, there are 3 steps:

    + The step 01 uses the first conda environment (example: ls01)
    
    + The step 02 and step 03 use the second conda environment as this is related to the python package 'osgeo' (example: ls02)
    
    + Three bash scripts of 'step01_gen_pkl.sh', 'step02_gen_csv.sh', 'step03_evaluate.sh' are used to run the step 01, the step 02, the step 03 respectively.
    
    + These bash scripts are currently written to run on server with SLURM manager. Users may need to modify to adapt users' server or local machine


2/ This source code is only for running and evaluating one image of 10980x10980x5 which matches the data provided in gAia project. However users can modify the code to run with different sizes of input image.The minimum size of the input image is 128x128x5


## Step 01: Use the pre-trained deep learning model to predict landslide and generate pkl file
  1/ Conda settings for the step 01:
  
     conda create --name ls01 python==3.8.0
     
     conda activate ls01
     
     pip install -r requirements_01.txt
     
     pip install protobuf==3.20.*

 2/ Input, output, and how to run the step 01:
 
  + Require the input directory (**./02_input_images/**) which contains the Sentinel-2 images with 5 bands (For an example: 33_T_UN_2018_3_S2A_33TUN_20180327_1_L2A_B02.tif, 33_T_UN_2018_3_S2A_33TUN_20180327_1_L2A_B03.tif, 33_T_UN_2018_3_S2A_33TUN_20180327_1_L2A_B04.tif, 33_T_UN_2018_3_S2A_33TUN_20180327_1_L2A_B08.tif, 33_T_UN_2018_3_S2A_33TUN_20180327_1_L2A_B11.tif)
  
  + Require the pretrained model at the directory (**01_pre_trained_models/model.h5**).  As the pretrained model is large (286 MB) and has not added into this github. Please contact the SBA (project lead) to get the model. Currently, it is available on Kronos server. 

  + The output **pkl** file will be automatically stored in the directory:  **'./11_output_pkl'**

  + Run the bash script to generate plk file:  **'step01_gen_pkl.sh'**

  + **NOTE**:  Estimate time for one image of 10980x10980x5 (7396 small image of 128x128x5): 10 minutes with gpu 2080 (11 GB) ---> need to use gpu for the step 01 as the pre-trained deep learning model is large and running with the batch size of 30 of 128x128x5.

## Step 02: Generate csv file from plk file
  1/ Conda setting for the step 02 (note that installing gdal depends on Conda version, the current version is **conda 22**):
  
     conda create --name ls02
     
     conda activate ls02
     
     conda install gdal
     
     pip install Pillow
     
     pip install shapely
     
     pip install -r requirements_02.txt


  + Require the **input plk file** available in the directory: **'./11_output_pkl'**

  + The **output csv file** will be automatically stored in the directory: **'./12_output_csv/'**.  The csv follow the standard format that the gAia project require without the header.

  + Run the scirp **'step02_gen_csv.sh'** for the step 02

## Step 03: Evaluate csv file
  - setup for step 03: use the same setup from the step 02(conda environment: ls02)

  - require the csv file available in the directory: ./12_output_csv

  - run the scirpt:
        - First, go to the directory with the commandline: cd ./31_evaluation/src
        - Second, run the bash scrip:  step03_evaluate.sh  > log
        - Finally, check the 'log' file


