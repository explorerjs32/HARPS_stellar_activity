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

def Sodium_Filters(wavelength):

    Na_D1 = 5895.92
    Na_D2 = 5889.95
    continuum_1 = 5805.0
    continuum_2 = 6090.0

    Na_filter_1 = (abs(wavelength - Na_D1) < 0.250).astype(float)
    Na_filter_2 = (abs(wavelength - Na_D2) < 0.250).astype(float)
    continuum_filter_1 = (abs(wavelength - continuum_1) < 5.000).astype(float)
    continuum_filter_2 = (abs(wavelength - continuum_2) < 10.000).astype(float)

    return Na_filter_1, Na_filter_2, continuum_filter_1, continuum_filter_2

def fit_continuum(wavelength, flux, order=5, upper_sigma=5, lower_sigma=3):
    # Manually mask the Sodium D lines
    mask = (abs(wavelength - 5895.924) > 6) & (abs(wavelength - 5889.950) > 6)

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

    global firstval

    shifted_flux, shifted_wave = pyasl.dopplerShift(target_wave, target_flux, v, edgeHandling="firstlast")
    
    return shifted_wave, shifted_flux

def Na_Index(flux, filter1, filter2, filter3, filter4):
        
    D1 = flux*filter1
    D2 = flux*filter2
    C1 = flux*filter3
    C2 = flux*filter4
    
    new_D1 = np.sort(D1[D1 != 0])
    new_D2 = np.sort(D2[D2 != 0])
    new_C1 = np.sort(C1)[::-1][0:10]
    new_C2 = np.sort(C2)[::-1][0:10]

    na_index = (np.mean(new_D1) + np.mean(new_D2))/(np.mean(new_C1) + np.mean(new_C2))

    return na_index

def Sigma_Clipping(data, sigma):
    isolated_data = []

    for i in range(len(data)):
        if data[i] >= np.median(data) - sigma*np.std(data) and data[i] <= np.median(data) + sigma*np.std(data):
            isolated_data.append(data[i])

    return np.asarray(isolated_data)

# Open the data
data = np.load('/media/fmendez/Seagate_Portable_Drive/Research/Research_Data/HARPS/HD26965_HARPS_data.npy').item()


indices = np.zeros(len(data['MJD']))
errors = np.zeros(len(data['MJD']))

for i in range(len(data['MJD'])):

    # Import the spectra of each observation
    star_wavelength = data['wave'][i][200000:234000]
    star_flux = data['flux'][i][200000:234000]

    # Plot The spectrum (Troubleshoot)
    '''
    plt.title(data['target'][i] + ' Spectrum - Observation %1i' %int(i+1))
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
    plt.title(data['target'][i] + ' SNR - Observation %1i' %int(i+1))
    plt.ylabel('SNR')
    plt.xlabel('Pixel Number')
    plt.plot(SNR, color='black')
    plt.show()
    '''
    # Generate the random spectra
    samples = 1
    test_index = []

    for j in range(samples):

        # Randoomly generate a new spectra for each observation
        test_flux = star_flux + np.random.randn(*star_flux.shape) * poisson_error

        # Plot the randomly generated spectrum (troubleshoot)
        '''
        plt.title(data['target'][i] + ' Randomly Generated Spectrum')
        plt.ylabel('Flux')
        plt.xlabel(r'Wavelength $\AA$')
        plt.plot(star_wavelength, test_flux)
        plt.show()
        '''
        # Fit the Continuum
        orig_flux = np.copy(test_flux)
        norm_flux = fit_continuum(star_wavelength, test_flux, order=5., upper_sigma=1, lower_sigma=0.1)    
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
        Filters = Sodium_Filters(star_wavelength)  

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
            filters = Sodium_Filters(wavelength)

            # Plot the aligned spectra with filters (troubleshoot)
            '''
            plt.title(data['target'][i] + ' Alligned Spectrum - Observation %1i' %int(i+1))
            plt.ylabel('Normalized Flux')
            plt.xlabel(r'Wavelength $\AA$')
            #plt.xlim(5888, 5898)
            plt.plot(wavelength, flux, color = 'black')
            plt.plot(wavelength, flux*filters[0], color = 'blue')
            plt.plot(wavelength, flux*filters[1], color = 'blue')
            plt.plot(wavelength, flux*filters[2], color = 'blue')
            plt.plot(wavelength, flux*filters[3], color = 'blue')
            plt.show()
            '''
            # Calculate the Na index for each of the generated spectra
            test_index.append(Na_Index(flux, filters[0], filters[1], filters[2], filters[3]))

        except (IndexError, UnboundLocalError, ValueError):
            pass

    # Isolate the data using sigma clipping
    isolated_index1 = Sigma_Clipping(test_index, 4)
    isolated_index2 = Sigma_Clipping(isolated_index1, 4)
    isolated_index3 = Sigma_Clipping(isolated_index2, 4)

    # Plot a histogram of the Gaussian distribution and sigma clipping of the indices (troubleshoot)
    '''
    plt.title('Observation %1i Na Index Gaussian Distribution' %int(i+1))
    plt.ylabel('Number of indices')
    plt.xlabel('Na Index')
    plt.hist(test_index, bins=30, color='black', histtype='step')
    plt.axvline(np.median(test_index), linestyle='--', color='green')
    plt.axvline(np.median(test_index) + (4.*np.std(test_index)), linestyle='--', color='red')
    plt.axvline(np.median(test_index) - (4.*np.std(test_index)), linestyle='--', color='red')
    #plt.show()
    fig =  plt.gcf()
    pdf.savefig(fig)
    plt.cla()

    plt.title('Observation %1i Na Index Sigma Clipping 1' %int(i+1))
    plt.ylabel('Number of indices')
    plt.xlabel('Na Index')
    plt.hist(isolated_index1, bins=30, color='black', histtype='step')
    plt.axvline(np.median(isolated_index1), linestyle='--', color='green')
    plt.axvline(np.median(isolated_index1) + (4.*np.std(isolated_index1)), linestyle='--', color='red')
    plt.axvline(np.median(isolated_index1) - (4.*np.std(isolated_index1)), linestyle='--', color='red')
    #plt.show()
    fig =  plt.gcf()
    pdf.savefig(fig)
    plt.cla()

    plt.title('Observation %1i Na Index Sigma Clipping 2' %int(i+1))
    plt.ylabel('Number of indices')
    plt.xlabel('Na Index')
    plt.hist(isolated_index2, bins=30, color='black', histtype='step')
    plt.axvline(np.median(isolated_index2), linestyle='--', color='green')
    plt.axvline(np.median(isolated_index2) + (4.*np.std(isolated_index2)), linestyle='--', color='red')
    plt.axvline(np.median(isolated_index2) - (4.*np.std(isolated_index2)), linestyle='--', color='red')
    #plt.show()
    fig =  plt.gcf()
    pdf.savefig(fig)
    plt.cla()

    plt.title(r'Observation %1i Na Index Sigma Clipping 3' %int(i+1))
    plt.ylabel('Number of indices')
    plt.xlabel('Na Index')
    plt.hist(isolated_index3, bins=30, color='black', histtype='step')
    plt.axvline(np.median(isolated_index3), linestyle='--', color='green')
    plt.axvline(np.median(isolated_index3) + np.std(isolated_index3), linestyle='--', color='red')
    plt.axvline(np.median(isolated_index3) - np.std(isolated_index3), linestyle='--', color='red')
    #plt.show()
    fig =  plt.gcf()
    pdf.savefig(fig)
    plt.cla()
    '''
    # Print the Index and error (troubleshhot)
        
    print '\nObservation:', i+1
    print 'N:', len(isolated_index3)
    print 'Index:', np.median(isolated_index3)
    print 'Error:', np.std(isolated_index3)

    indices[i] = np.median(np.asarray(isolated_index3))
    errors[i] = np.std(np.asarray(isolated_index3))
        
# Save the data
save_data = np.column_stack([indices, errors, np.asarray(data['MJD'])])
print save_data
np.savetxt('HD26965_Na_Variability.dat', save_data)

# Plot the Ha index variability

plt.title(data['target'][i] + ' Na Index Variability')
plt.xlabel('BJD')
plt.ylabel('Na Index')
plt.errorbar(data['MJD'], indices, yerr = errors, fmt = 'ko', capsize = 3)
plt.show()    

























