import numpy as np
import scipy.constants as sc
import sys

import healpy as hp
from astropy.io import fits

from time import time

from dipolefunctions_CatWISE import *
from helperfunctions import *

def main(N,simnum,vel,seed,galcut,direction='CMB') :
	
	print('Running {:0.0f} simulations of initially {:0.0f} sources each for a velocity of {:0.0f} km/s'.format(simnum,N,vel/1000.),flush=True)
	
	catwise_extended = fits.open('../Data/catwise_agns_masked_final_w1lt16p5_alpha.fits')[1].data
	
	flux_w1 = catwise_extended['k']*catwise_extended['nu_W1_iso']**catwise_extended['alpha_W1']
	alpha = -catwise_extended['alpha_W1']
	
	W1_fluxcut = 8.52707e-28
	N_output = len(flux_w1[(flux_w1>W1_fluxcut)])
	dlon,dlat = -121.78,28.80
	
	print('Reading and making mask')
	
	masks = fits.open('../Data/MASKS_exclude_master_final.fits')[1].data
	nside_hi = 1024
	nside_lo = 64
	galcut = 30
	mask = makeMask(masks,galcut=galcut,nside=nside_hi,masking='onesided')
	mask = hp.ud_grade(mask,nside_out=nside_lo)
	mask[(mask!=1)] = 0
	
	print('Beginning simulations')
	
	if direction=='CMB' :
		lonlats_dipole,lonlats_dipole_corrected,d,n = doAll_Vectors_Sim_resampling(N, N_output, simnum, alpha, flux_w1, lon_psmask=[lon_LMC,lon_SMC], lat_psmask=[lat_LMC,lat_SMC], rad_mask=[rad_LMC,rad_SMC], vel=vel, seed=seed, galcut=galcut, do_resampling=True, W1_fluxcut=W1_fluxcut, estimator='healpy', masking='onesided', nside=nside_lo, mask=mask)
	elif direction=='CW' :
		lonlats_dipole,lonlats_dipole_corrected,d,n = doAll_Vectors_Sim_resampling(N, N_output, simnum, alpha, flux_w1, lon_direction=dlon, lat_direction=dlat, lon_psmask=[lon_LMC,lon_SMC], lat_psmask=[lat_LMC,lat_SMC], rad_mask=[rad_LMC,rad_SMC], vel=vel, seed=seed, galcut=galcut, do_resampling=True, W1_fluxcut=W1_fluxcut, estimator='healpy', masking='onesided', nside=nside_lo, mask=mask)
	
	print('Completed.  Writing results to files.',flush=True)
	
	np.savetxt('../SavedData/CatWISE_DipoleSimulations_{:0.0f}sourcesInp_{:0.0f}sourcesOut_{:0.0f}kms_{:0.0f}sims_lonlats_run{:0.0f}_{:s}dir.txt'.format(N,N_output,vel/1000.,simnum,seed,direction),np.array(lonlats_dipole).transpose(),header='Longitude and latitude of uncorrected dipole directions')
	
	np.savetxt('../SavedData/CatWISE_DipoleSimulations_{:0.0f}sourcesInp_{:0.0f}sourcesOut_{:0.0f}kms_{:0.0f}sims_lonlats_corrected_run{:0.0f}_{:s}dir.txt'.format(N,N_output,vel/1000.,simnum,seed,direction),np.array(lonlats_dipole_corrected).transpose(),header='Longitude and latitude of corrected dipole directions')
	
	np.savetxt('../SavedData/CatWISE_DipoleSimulations_{:0.0f}sourcesInp_{:0.0f}sourcesOut_{:0.0f}kms_{:0.0f}sims_d_run{:0.0f}_{:s}dir.txt'.format(N,N_output,vel/1000.,simnum,seed,direction),d,header='Dipole amplitudes (biased)')
	
	np.savetxt('../SavedData/CatWISE_DipoleSimulations_{:0.0f}sourcesInp_{:0.0f}sourcesOut_{:0.0f}kms_{:0.0f}sims_n_run{:0.0f}_{:s}dir.txt'.format(N,N_output,vel/1000.,simnum,seed,direction),n,header='Number of sources of final sample')
	
if __name__ == '__main__' :
	
	if len(sys.argv)<6 :
		raise ValueError('Please provide the five input arguments, N, simnum, vel, seed, and direction.')
	
	N = int(sys.argv[1])
	simnum = int(sys.argv[2])
	vel = float(sys.argv[3])
	seed = int(sys.argv[4])
	galcut = float(sys.argv[5])
	direction = sys.argv[6]
	
	main(N,simnum,vel,seed,galcut,direction=direction)
