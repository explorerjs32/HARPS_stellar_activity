from astropy.io import fits
import numpy as np
import scipy.interpolate
import scipy.optimize
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyAstronomy import pyasl
import glob
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', np.RankWarning)

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

def Ha_Filters(wavelength):

    Ha = 6562.808
    continuum_1 = 6550.87
    continuum_2 = 6580.31

    Ha_filter = (abs(wavelength - Ha) < 0.800).astype(float)
    continuum_filter_1 = (abs(wavelength - continuum_1) < 5.375).astype(float)
    continuum_filter_2 = (abs(wavelength - continuum_2) < 4.375).astype(float)

    return Ha_filter, continuum_filter_1, continuum_filter_2

def fit_continuum(wavelength, flux, order=5, upper_sigma=5, lower_sigma=3):
    # Manually mask the Ha line
    mask = (abs(wavelength - 6562.808) > 6)

    while True:
        p = np.polyfit(wavelength[mask], flux[mask], order)
        continuum = np.polyval(p, wavelength[mask])

        separation = abs(flux[mask] - continuum)
        sigma = np.std(separation)
        new_mask = ((flux[mask] > continuum) & (separation < upper_sigma * sigma)) | ((flux[mask] < continuum) & (separation < lower_sigma * sigma))

        if np.all(new_mask) or np.sum(new_mask) <= order:
            break
        else:
            mask[mask] = new_mask

    return flux/np.polyval(p, wavelength)

def ccf_eval(rv, target_wave, target_flux, tck):
    c = 2.99792485e8
    shifted_wave = target_wave * np.sqrt((1+rv/c)/(1-rv/c))
    shifted_flux = scipy.interpolate.splev(shifted_wave, tck)
    return np.sum((shifted_flux - target_flux)**2)


def align_spectrum(template_wave, template_flux, target_wave, target_flux,
            rv_bounds=(-1e5, 1e5), tol=1e-5, coarse_samples=100, verbose=False):
   
    rv_bounds = sorted(rv_bounds)

    # Remove nan values from data
    mask = np.isfinite(target_flux) & np.isfinite(template_flux)
    target_flux = target_flux[mask]
    target_wave = target_wave[mask]

    # Make sure both spectra are arranged so that the wavelength values are ascending
    inds = np.argsort(target_wave)
    target_wave = target_wave[inds]
    target_flux = target_flux[inds]

    inds = np.argsort(template_wave)
    template_wave = template_wave[inds]
    template_flux = template_flux[inds]

    # Clip target data so that we do not run out of template
    c = 2.99792485e8
    low_wave = target_wave[0] * np.sqrt((1+rv_bounds[1]/c)/(1-rv_bounds[1]/c))
    high_wave = target_wave[-1] * np.sqrt((1+rv_bounds[0]/c)/(1-rv_bounds[0]/c))

    mask = (low_wave < target_wave) & (target_wave < high_wave)

    target_wave = target_wave[mask]
    target_flux = target_flux[mask]

    # Build the spline for wavelength shifting
    tck = scipy.interpolate.splrep(template_wave, template_flux)

    coarse_samples = 100

    if coarse_samples > 0:
        # Coarsely sample the CCF to find the primary peak
        rv_coarse = np.linspace(rv_bounds[0], rv_bounds[1], coarse_samples)
        ccf_coarse = np.empty(rv_coarse.shape)
        for i, rv in enumerate(rv_coarse):
            ccf_coarse[i] = ccf_eval(rv, target_wave, target_flux, tck)
        '''
        plt.plot(rv_coarse, ccf_coarse)
        plt.show()
        '''

        ind = ccf_coarse.argmin()
        
        bracket = (rv_coarse[ind-1], rv_coarse[ind+1])
        tol *= 1/abs(rv_coarse[ind])
        
        if verbose:
            print('Optimizing between {} m/s and {} m/s'.format(*bracket))
        
    else:
        bracket = rv_bounds
        tol *= abs(1/np.mean(rv_bounds))

    result = scipy.optimize.minimize_scalar(ccf_eval, bracket=bracket, tol=tol,
                             args=(target_wave, target_flux, tck))

    rv = np.linspace(start=result.x - 0.01, stop=result.x + 0.01, num=1000)
    ccf = np.empty(rv.shape)
    for i, r in enumerate(rv):
        ccf[i] = ccf_eval(r, target_wave, target_flux, tck)

    # Compute a little trendline through the peak to help the eye follow the noisy data
    p = np.polyfit(rv, ccf, 5)
    smooth = np.polyval(p, rv)
    
    
    #print n, (result.x/2.)*(1.e-3)
    #print(abs(result.x - rv[smooth.argmin()]))
    '''
    plt.xlabel('RV (m/s)')
    plt.ylabel('CCF value')
    plt.plot(rv, ccf)
    plt.plot(rv, smooth)
    plt.plot(result.x, result.fun, 'ko')
    plt.plot(rv[smooth.argmin()], smooth.min(), 'ro')
    plt.show()
    '''
   
    result.x = rv[smooth.argmin()]
    v = (result.x/2.)*(1.e-3)

    shifted_flux, shifted_wave = pyasl.dopplerShift(target_wave, target_flux, v, edgeHandling="firstlast")
    
    return shifted_wave, shifted_flux

def Ha_Index(flux, filter1, filter2, filter3):

    # Place the filters on the Ha line and the continuum 
    Ha = flux*filter1
    C1 = flux*filter2
    C2 = flux*filter3

    # Select only the flux values withing the filters
    new_Ha = np.sort(Ha[Ha != 0.])
    new_C1 = np.sort(C1[C1 != 0.])
    new_C2 = np.sort(C2[C2 != 0.])

    # Calculate the Ha index
    Ha_index = np.mean(new_Ha)/(0.5*(np.mean(new_C1) + np.mean(new_C2)))

    return Ha_index

def Sigma_Clipping(data, sigma):
    isolated_data = []

    for i in range(len(data)):
        if data[i] >= np.median(data) - sigma*np.std(data) and data[i] <= np.median(data) + sigma*np.std(data):
            isolated_data.append(data[i])

    return np.asarray(isolated_data)

# Extract the data from the fits files
wavelengths = []
fluxes = []
target_names = []
BJDs_string = []
BJDs_float = []
MJDs = []

file_list = glob.glob('HARPS*fits')

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
    BJDs_string.append(str(BJD))
    BJDs_float.append(float(BJD))
    MJDs.append(MJD)

    print 'Data Appended Observation', i+1

indices = np.zeros(len(fluxes))
errors = np.zeros(len(fluxes))

for i in range(len(MJDs)):
    # Import the Wavelength and the Flux for each observation
    star_wavelength = wavelengths[i][276000:281000]
    star_flux = fluxes[i][276000:281000]

    # Plot The spectrum (Troubleshoot)
    '''
    plt.title(target_names[i] + ' Spectrum - Observation %1i' %int(i+1))
    plt.ylabel('Flux')
    plt.xlabel(r'Wavelength $\AA$')
    plt.plot(star_wavelength, star_flux, color='black')
    plt.show()
    '''
    # Calculate the Poisson error and the SNR corresponding to each flux 
    poisson_error = np.sqrt(0.71*star_flux)
    SNR = (0.71*star_flux)/poisson_error

    # Plot the SNR (Troubleshoot)
    '''
    plt.title(target_names[i] + ' SNR - Observation %1i' %int(i+1))
    plt.ylabel('SNR')
    plt.xlabel('Pixel Number')
    plt.plot(SNR, color='black')
    plt.show()
    '''    
    # Generate the random spectra
    samples = 1000
    test_index = []

    for j in range(samples):
        # Randoomly generate a new spectra for each observation
        test_flux = star_flux + np.random.randn(*star_flux.shape) * poisson_error

        # Plot the randomly generated spectrum (troubleshoot)
        '''
        plt.title(target_names[i] + ' Randomly Generated Spectrum')
        plt.ylabel('Flux')
        plt.xlabel(r'Wavelength $\AA$')
        plt.plot(star_wavelength, test_flux)
        plt.show()
        '''     
        # Fit the Continuum
        try:
            orig_flux = np.copy(test_flux)
            norm_flux = fit_continuum(star_wavelength, test_flux, order=2., upper_sigma=1, lower_sigma=0.1)    
            continuum = orig_flux/norm_flux

            # Plot the fit continuum and the normalized spectra (troubleshoot)
            '''
            plt.title('Observation %1i Fit Continuum' %int(i+1))
            plt.ylabel('Flux')
            plt.xlabel(r'Wavelength $\AA$')
            plt.plot(star_wavelength, orig_flux, color = 'salmon')
            plt.plot(star_wavelength, continuum, '--', color = 'blue')
            plt.show()
        
            plt.title('Observation %1i Normalized Spectrum' %int(i+1))
            plt.ylabel('Normalized Flux')
            plt.xlabel(r'Wavelength $\AA$')
            plt.plot(star_wavelength, norm_flux, color = 'black')
            plt.show()
            '''
        except IndexError:
            pass
        
        Filters = Ha_Filters(star_wavelength)  

        # Plot the filters (troubleshoot)
        '''
        plt.title('Observation %1i' %int(i+1))
        plt.ylabel('Normalized Flux')
        plt.xlabel(r'Wavelength $\AA$')
        plt.xlim(5888, 5898)
        plt.axvline(5889.95, color='red')
        plt.axvline(5895.92, color='red')
        plt.plot(star_wavelength, norm_flux, color = 'black')
        plt.plot(star_wavelength, norm_flux*Filters[0], 'b')
        plt.plot(star_wavelength, norm_flux*Filters[1], 'b')
        plt.plot(star_wavelength, norm_flux*Filters[2], 'b')
        plt.plot(star_wavelength, norm_flux*Filters[3], 'b')
        plt.show()
        '''
        
        try:

            if i == 0:
                # Create the template spectrum
                flux, wavelength = pyasl.dopplerShift(star_wavelength, norm_flux, 21., edgeHandling="firstlast")
                template_wavelength, template_flux = np.copy(wavelength), np.copy(flux)

            else:
                # Align spectra and error with the template
                wavelength, flux = align_spectrum(template_wavelength, template_flux, star_wavelength, norm_flux)

            # Calcuate the new filters with the aligned wavelength
            filters = Ha_Filters(wavelength)

            # Plot the aligned spectra with filters (troubleshoot)
            ''' 
            plt.title(target_names[i] + ' Alligned Spectrum - Observation %1i' %int(i+1))
            plt.ylabel('Normalized Flux')
            plt.xlabel(r'Wavelength $\AA$')
            #plt.xlim(5888, 5898)
            plt.plot(wavelength, flux, color = 'black')
            plt.plot(wavelength, flux*filters[0], color = 'blue')
            plt.plot(wavelength, flux*filters[1], color = 'blue')
            plt.plot(wavelength, flux*filters[2], color = 'blue')
            plt.show()
            '''
            # Calculate the Na index for each of the generated spectra
            test_index.append(Ha_Index(flux, filters[0], filters[1], filters[2]))

        except (IndexError, UnboundLocalError, ValueError):
            pass

    # Isolate the data using sigma clipping
    isolated_index1 = Sigma_Clipping(test_index, 4)
    isolated_index2 = Sigma_Clipping(isolated_index1, 4)
    isolated_index3 = Sigma_Clipping(isolated_index2, 4)
    
    # Print the Index and error (troubleshhot)
    print '\nObservation:', i+1
    print 'MJD:', int(MJDs[i])
    print 'N:', len(isolated_index3)
    print 'Index:', np.median(isolated_index3)
    print 'Error:', np.std(isolated_index3)

    indices[i] = np.median(np.asarray(isolated_index3))
    errors[i] = np.std(np.asarray(isolated_index3))

# Save the data
save_data = np.column_stack([indices, errors, np.asarray(MJDs)])
np.savetxt('HD26965_Ha_Variability.dat', save_data)

# Plot the Ha index variability
plt.title('Ha Index Variability')
plt.xlabel('BJD')
plt.ylabel('Ha Index')
plt.errorbar(MJDs, indices, yerr = errors, fmt = 'ko', capsize = 3)
plt.show()    


















