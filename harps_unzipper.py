import tarfile as tar
import numpy as np
import os

###################################################################        EXTRACT THE FITS FILES        #####################################################
# Create the path to the tar files and list them
file_path = '/media/fmendez/Seagate_Portable_Drive/Research/Research_Data/HARPS/zip_files/HD26965'
file_list = os.listdir(file_path)

# Extract the fits file from the tar files
for i in file_list:
    print (i)
    tar_file = tar.open(i)
    intar_file_list = tar_file.getnames()

    #print ('Extracting', intar_file_list[-2], 'from', i)


        
