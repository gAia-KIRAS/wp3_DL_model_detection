import pickle
import os
#import glob
import numpy as np
from PIL import Image
import sys
import csv
import time

#--- for inferring lat/logitude
from osgeo import gdal, osr
import glob
import pickle
import geopandas as gpd
from numba import jit
import numpy as np
import PIL
from PIL import Image, TiffImagePlugin
from shapely.geometry import Point, Polygon, box

from polygon_02 import *

#------------------------------------------- Source code from the Internet:  https://stackoverflow.com/questions/63004971/find-latitude-longitude-coordinates-of-every-pixel-in-a-geotiff-image
def pixel2coord(img_path, x, y):
    """
    Returns latitude/longitude coordinates from pixel x, y coords

    Keyword Args:
      img_path: Text, path to tif image
      x: Pixel x coordinates. For example, if numpy array, this is the column index
      y: Pixel y coordinates. For example, if numpy array, this is the row index
    """
    # Open tif file
    ds = gdal.Open(img_path)

    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(ds.GetProjectionRef())

    # create the new coordinate system
    # In this case, we'll use WGS 84
    # This is necessary becuase Planet Imagery is default in UTM (Zone 15). So we want to convert to latitude/longitude
    wgs84_wkt = """
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]]"""
    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)

    # create a transform object to convert between coordinate systems
    transform = osr.CoordinateTransformation(old_cs,new_cs) 
    
    gt = ds.GetGeoTransform()

    # GDAL affine transform parameters, According to gdal documentation xoff/yoff are image left corner, a/e are pixel wight/height and b/d is rotation and is zero if image is north up. 
    xoff, a, b, yoff, d, e = gt

    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff

    lat_lon = transform.TransformPoint(xp, yp) 

    xp = lat_lon[0]
    yp = lat_lon[1]
    
    return (xp, yp)


def find_img_coordinates(img_array, image_filename):
    img_coordinates = np.zeros((img_array.shape[0], img_array.shape[1], 2)).tolist()
    for row in range(0, img_array.shape[0]):
        for col in range(0, img_array.shape[1]): 
            img_coordinates[row][col] = Point(pixel2coord(img_path=image_filename, x=col, y=row))
    return img_coordinates


def find_image_pixel_lat_lon_coord(image_filenames, output_filename):
    """
    Find latitude, longitude coordinates for each pixel in the image

    Keyword Args:
      image_filenames: A list of paths to tif images
      output_filename: A string specifying the output filename of a pickle file to store results

    Returns image_coordinates_dict whose keys are filenames and values are an array of the same shape as the image with each element being the latitude/longitude coordinates.
    """
    image_coordinates_dict = {}
    for image_filename in image_filenames:
        print('Processing {}'.format(image_filename))
        img = Image.open(image_filename)
        img_array = np.array(img)
        img_coordinates = find_img_coordinates(img_array=img_array, image_filename=image_filename)
        image_coordinates_dict[image_filename] = img_coordinates
        with open(os.path.join(DATA_DIR, 'interim', output_filename + '.pkl'), 'wb') as f:
            pickle.dump(image_coordinates_dict, f)
    return image_coordinates_dict

#----------------------------------------------------------------------------------------------------------------


#------------------------------------------- Main Function Here
org_image_size = 10980
#tif_data_dir = '/var/data/storage/datasets/image/01_AI_for_Earth/02_slandslide/04_data_gaia_S2/'
tif_data_dir = './02_input_images'

ifolder_pkl = './11_output_pkl'
file_list   = os.listdir(ifolder_pkl)

#------ For each pkl file
for ifile in file_list:
    ifile_dir = os.path.join(ifolder_pkl,ifile)

    with open(ifile_dir, 'rb') as f:
        data = pickle.load(f)
        
        if len(data) >= 1: #the pkl file contains the landsliding
            file_name = ifile.split('.pkl')[0]

            info_tiles = file_name.split('_')[-5] 
            info_time  = file_name.split('_')[-4] 
            info_year  = info_time[0:4]
            info_day   = info_time[0:4]+'-'+info_time[4:6]+'-'+info_time[6:8]
        
            content = []
            for i_data in range(0, len(data)):

                #--- extract local reagion
                local_row  = data[i_data][0]
                local_col  = data[i_data][1]
                local_mask = data[i_data][2]
                real_mask  = data[i_data][3]

                #modify timestamp format
                info_timestamp  = data[i_data][4]
                info_timestamp_day  = info_timestamp[0:4]+'-'+info_timestamp[4:6]+'-'+info_timestamp[6:8]
                info_timestamp_hour = info_timestamp[9:11]+':'+info_timestamp[11:13]+':'+info_timestamp[13:]
                info_timestamp = info_timestamp_day+' '+info_timestamp_hour

                if local_row >= 86  or local_col >= 86:
                    print('===================== ERROR 01: Row/col is larger than 85,  File name:', file_name, 'row:', local_row, 'col:', local_col)
                    exit()
                
                for i in range(0,128):
                    for j in range(0,128):
                        if local_mask[i,j] == 1:
                            #print(i, j, local_row, local_col)

                            #--- Infer global row, col (global row/col in the big images) from the local row/col (local row/col from 128x128 images)
                            if local_row == 85:
                                global_row = org_image_size - 128 + i
                            else:    
                                global_row = local_row*128 + i

                            if local_col == 85:    
                                global_col = org_image_size - 128 + j
                            else:    
                                global_col = local_col*128 + j

                            #--- get probability at the pixel
                            info_pred_prob = real_mask[i,j]


                            #---- Infer the Lat/Longitude postions from row and col
                            full_file_name  = ifile.split('.')[0]+'B02.tif'
                            #tif_file_dir    = os.path.join(tif_data_dir, info_year, info_tiles, 'tmp', full_file_name)
                            tif_file_dir    = os.path.join(tif_data_dir, full_file_name)
                            lat, lon  = pixel2coord(tif_file_dir, float(global_col), float(global_row))

                            #----- Add by Lam Pham to check if the position (lat,lon) is in AOI 
                            my_pol_AOI = my_pol_02
                            lat_ck_02 = float(lat)
                            lon_ck_02 = float(lon)

                            point_ck = shapely.geometry.Point(lon_ck_02, lat_ck_02)
                            is_true = point_ck.within(my_pol_AOI)

                            #---- append all info into list
                            if is_true:
                                content.append(['Version_v1.1', info_tiles, info_year, global_row, global_col, info_day, info_pred_prob, info_timestamp, lat, lon]) #without threshold

                            
            #--- Save file
            if len(content) == 0:  # Add by Lam Pham to check if the position (lat,lon) is in AOI
                print('ERROR: The lat/lontitude with landslide detections are not in AOI --> No CSV file is generated --> EXIT')
                exit()
            else:
                if not os.path.exists('12_output_csv'):
                    os.makedirs('12_output_csv')

                store_dir = './12_output_csv'
                if not os.path.exists(store_dir):
                    os.makedirs(store_dir)

                csv_res_file = os.path.join(store_dir, file_name+'.csv')
                with open(csv_res_file, 'w') as f:
                    # create the csv writer
                    writer = csv.writer(f)
                    #add header
                    #writer.writerow(['version', 'threshold','tile','year','row','column', 'date', 'probability', 'timestamp', 'lat', 'lon'])  #with threshold
                    writer.writerow(['version', 'tile','year','row','column', 'date', 'probability', 'timestamp', 'lat', 'lon'])   #without threshold
                
                    # write a row to the csv file
                    for i_cont in content:
                        writer.writerow(i_cont)
                    #exit()

               
