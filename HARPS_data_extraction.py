import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob

##############################################################        EXTRACT THE SPECTRA FROM THE FITS FILES        #########################################
def Calculated_Wavelength(flux, start, step):
    start_wave = start
    wavelength = np.zeros(len(flux))

    for i in range(len(flux)):
        if i == 0:
            wavelength[i] = start_wave

        else:
            wavelength[i] = start_wave + step
            start_wave = np.copy(wavelength[i])

    return wavelength

# Create the file list
target = 'HD26965A'

data_files = '/media/fmendez/Seagate_Portable_Drive/Research/Research_Data/HARPS/'
file_list = glob.glob(data_files + 'fits_files/' + target + '/*.fits')

# Extract the data from the fits files
wavelengths = []
fluxes = []
target_names = []
MJDs = []

for i in range(len(file_list)):

    hdul = fits.open(file_list[i])

    target = fits.open(file_list[i])[0].header['HIERARCH ESO OBS TARG NAME']
    BJD = fits.open(file_list[i])[0].header['HIERARCH ESO DRS BJD']
    MJD = fits.open(file_list[i])[0].header['MJD-OBS']
    flux = fits.getdata(file_list[i])
    
    if i == 0:
        
        start_wave = hdul[0].header['CRVAL1']
        step_wave = hdul[0].header['CDELT1']
        wavelength = Calculated_Wavelength(flux, start_wave, step_wave)
 
    wavelengths.append(wavelength)
    fluxes.append(flux)
    target_names.append(target)
    MJDs.append(MJD)

    print ('Observation %i done' %int(i+1))
print (len(wavelengths))
# Create and save the dictionary of the data
HD26965_HARPS_data = {}

HD26965_HARPS_data['wave'] = wavelengths
HD26965_HARPS_data['flux'] = fluxes
HD26965_HARPS_data['target'] = target_names
HD26965_HARPS_data['MJD'] = MJDs

np.savez(data_files + 'HD26965A_HARPS_data', wave=wavelengths, flux=fluxes, target=target_names, MJD=MJDs)










    

