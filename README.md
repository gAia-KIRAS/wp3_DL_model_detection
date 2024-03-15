# WP3: Deep Learning Model For LandSlide Detection using Sentinel-2 Remote Sensing Image

I/ General

1/ To run this model, there are 3 steps:

    + The step 01 uses the first conda environment (example: ls01)
    
    + The step 02 and step 03 use the second conda environment as this is related to the python package 'osgeo' (example: ls02)
    
    + Three bash scripts of 'step01_gen_pkl.sh', 'step02_gen_csv.sh', 'step03_evaluate.sh' are used to run the step 01, the step 02, the step 03 respectively.
    
    + These bash scripts are currently written to run on server with SLURM manager. Users may need to modify to adapt users' server or local machine


2/ The code is only for running and evaluating one sample (one large image of 10980x10980x5). Users need to modify the code to run massively.


## Running ## 
#----- step 01
+ Step 01: Use pre-trained deep learning model to predict and generate pkl file
  - setup for step 01:
     conda create --name ls01 python==3.8.0
     conda activate ls01
     pip install -r requirements_01.txt
     pip install protobuf==3.20.*

  - require the input directory (./02_input_images/) which contains the Sentinel-2 images

  - the output pkl file will be stored in the directory:  ./11_output_pkl

  - run the bash script to generate plk file:  step01_gen_pkl.sh

  - estimate time for one big image of 10980x10980x5 (7396 small image of 128x128x5): 10 minutes with gpu 2080 (11 GB) a
    ---> need to use gpu for step 01 as the pre-trained DL model is large and running with the batch size of 30 of 128x128x5.

#----- step 02
+ Step 02: Generate csv file from plk file
  - setup for step 02 (note that installing gdal depends on Conda version, the current version is conda 22):
     conda create --name ls02
     conda activate ls02
     conda install gdal
     pip install Pillow
     pip install shapely
     pip install -r requirements_02.txt


  - require the plk file available in the directory: ./11_output_pkl

  - the output csv file will be stored in the directory: ./12_output_csv/

  - run the scirpt: step02_gen_csv.sh

#----- step 03
+ Step 03: Evaluate csv file
  - setup for step 03: use the same setup from the step 02(conda environment: ls02)

  - require the csv file available in the directory: ./12_output_csv

  - run the scirpt:
        - First, go to the directory with the commandline: cd ./31_evaluation/src
        - Second, run the bash scrip:  step03_evaluate.sh  > log
        - Finally, check the 'log' file


