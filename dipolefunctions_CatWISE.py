import numpy as np
import scipy.constants as sc
from lmfit import Minimizer, Parameters
import healpy as hp
#from time import time

lon_CMBdipole = 264.021
lat_CMBdipole = 48.253
velocity_CMBframe = 369000.

lon_LMC,lat_LMC = 280.46526218382076, -32.888503352167234
rad_LMC = 11.5
lon_SMC,lat_SMC = 302.7969909022401, -44.29931060764203
rad_SMC = 5.519999980926514

deg2rad = np.pi / 180.
rad2deg = 180. / np.pi

W1_fluxcut = 1

def dir2vec(lon, lat) :
	
	"""
	Converts longitude and latitude in degrees into a Cartesian vector with unit length
	"""
	
	theta,phi = deg2rad*(90.-lat),deg2rad*lon
	
	ct = np.cos(theta)
	st = np.sin(theta)
	cp = np.cos(phi)
	sp = np.sin(phi)
	
	vec = np.array([st*cp,st*sp,ct])
	
	return vec


def vec2dir(vec) :
	
	"""
	Converts a Cartesian vector with unit length into longitude and latitude in degrees
	"""
	
	x = vec[0]
	y = vec[1]
	z = vec[2]
	
	theta = np.arccos(z)
	phi = np.arctan2(y,x)
	
	lat,lon = 90.-rad2deg*theta,rad2deg*phi
	
	return lon,lat


def angdist(vec1, vec2) :
	
	"""
	Computes the angular distance of two Cartesian vectors with unit length in degrees
	"""
	
	angle = rad2deg * np.arccos(np.dot(vec1,vec2))
	
	return angle


def scattomap(lon, lat, nside=16):
	
	""" Returns a histogram of celestial objects whose position is given in latitute and longitude by bins chosen by HEALPix (at resolution nside) """
	
	hmap = np.zeros(hp.nside2npix(nside))
	hmap = hmap + np.bincount(hp.ang2pix(nside,lon,lat,lonlat=1), minlength=hp.nside2npix(nside))
	
	return hmap


def getRotationMatrix_Z(lon=lon_CMBdipole) :
	
	"""
	Computes the rotation matrix around the Z-axis for moving a vector with longitude lon to the zero meridian
	"""
	
	a1 = -lon
	
	c1 = np.cos(deg2rad * a1)
	s1 = np.sin(deg2rad * a1)
	
	rot_mat = np.array([[c1, -s1, 0],[s1, c1, 0.],[0, 0, 1.]])
	
	return rot_mat


def rotateVectors(vec, rot_mat) :
	
	"""
	Given a rotation matrix, computes the rotated vector of an input Cartesian vector with unit length
	"""
	
	vec_rot = np.matmul(rot_mat,vec)
	
	return vec_rot


def mag2flux(mag, band='W1', fc=0) :
	
	"""
	Converts magnitude in a WISE band to flux in Jy
	
	fc is a correction factor dependent on the measured spectrum
	"""
	
	if band == 'W1' : f0 = 309.54
	elif band == 'W2' : f0 = 171.787
	
	f = f0 * 10**(-mag/2.5)
	
	if fc :
		f/fc
	
	return f


def getDipoleVectors_Crawford(vec) :
	
	
	"""
	Computes the preferred direction and the estimated dipole amplitude from a sample of vectors
	
	This is the linear estimator as employed by Crawford
	"""
	
	num = float(len(vec[0,:]))
	
	dipole = np.sum(vec,axis=1)
	norm = np.sqrt(np.dot(dipole,dipole))
	dipole_norm = dipole/norm
	
	d = 3./num * norm
	
	return dipole_norm,d


def getDipoleVectors_healpy(densitymap, mask=[None], galcut=0, verbose=False) :
	
	
	"""
	Computes the preferred direction and the estimated dipole amplitude from a density map
	
	This is a wrapper for the healpy routine fit_dipole
	"""
	
	if mask[0] != None :
		densitymap[(mask == 0)] = np.nan
	
	residual,monopole,dipole = hp.remove_dipole(densitymap,bad=np.nan,fitval=True,gal_cut=galcut,verbose=verbose)
	norm = np.sqrt(np.dot(dipole,dipole))
	dipole_norm = dipole/norm
	
	d = norm/monopole
	
	return dipole_norm,d


def getDipoleVectors_quadratic(densmap, weights=None, nside=32, mask=[None]):
    
	if mask[0]==None :
		mask = np.ones_like(densmap)
		mask[(densmap==0)]=0
    
	nonzer=[(mask==1)]
	
	def SumToMinimize(pars, outputresiduals=False):
		
		Nbar, A, lon, lat = pars[0], pars[1], pars[2], pars[3]
		pvec = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))
		vec = dir2vec(lon, lat)
		costhetap = np.cos(angdist(vec, pvec)*deg2rad)
		tos = (np.power( (densmap - Nbar*(1. +  A*costhetap)), 2 )/(Nbar*(1. +  A*costhetap)))
		if outputresiduals :
			return np.sum(tos*nonzer), densmap - Nbar*(1. +  A*costhetap) #1810.04960 Eq 17
		else :
			return np.sum(tos*nonzer)
	
	def lmfittomin(lmpars):
		sppars = list(lmpars.valuesdict().values())
		return SumToMinimize(sppars)
		
	lmp = Parameters()
	
	Nbguess = np.sum(densmap[nonzer])/float(len(densmap[nonzer]))
	initlon,initlat = hp.pix2ang(nside,np.random.choice(np.arange(hp.nside2npix(nside))), lonlat=True)
	inguess = np.array([Nbguess, 0.011, initlon, initlat])
	
	pnames = ['Nbar', 'A', 'DipLon', 'DipLat']
	bnds = ((Nbguess*0.7, Nbguess*1.5),(0, 1.0), (0,360.),(-90.0,90.0))
	
	for val, name, bnd in zip(inguess, pnames, bnds):
		lmp.add(name, value=val, min=bnd[0], max=bnd[1])
	
	minner = Minimizer(lmfittomin, lmp)
	resquad = minner.minimize(method = 'ampgo')
	
	mfval, residualmap = SumToMinimize([resquad.params['Nbar'].value, resquad.params['A'].value, resquad.params['DipLon'].value, resquad.params['DipLat'].value],outputresiduals=True)
	
	vec_dipole = dir2vec(resquad.params['DipLon'].value,resquad.params['DipLat'].value)
	d = resquad.params['A'].value
	
	return vec_dipole,d



def dip2vel(d, x=1., alpha=0.75) :
	
	"""
	Converts dipole amplitude to velocity beta according to Ellis+Baldwin
	"""
	
	vel = d * sc.c / (2.+x*(1.+alpha))
	
	return vel

def vel2dip(vel, x=1., alpha=0.75) :
	
	"""
	Converts velocity to dipole amplitude according to Ellis+Baldwin
	"""
	
	d = vel / sc.c * (2.+x*(1.+alpha))
	
	return d


def correctDirection(dipole_norm, galcut=30.) :
	
	"""
	Corrects the bias in an estimate of dipole direction of a sample of vectors in case the Galactic plane is masked
	
	The calculation of the bias B follows Rubart with small modifications.  His expression of B is
	
		B = (1-1.5*np.cos(alpha)+0.5*np.cos(alpha)**3) / (1-np.cos(alpha)**3)
	
	The two expressions are equivalent.
	"""
	
	lon,lat = vec2dir(dipole_norm)
	rotmat = getRotationMatrix_Z(lon)
	rotmat_inv = np.linalg.inv(rotmat)
	
	
	dipole_norm_rot = np.matmul(rotmat,dipole_norm)
	lon_rot,lat_rot = vec2dir(dipole_norm_rot)
	
	alpha = 90. - galcut
	alpha = deg2rad * alpha
	
	if lat_rot > 0 :
		theta = 90. - lat_rot
	else :
		theta = 90. + lat_rot
	
	theta = deg2rad * theta
	
	
	B = (1.-1./8.*(9.*np.cos(alpha)-np.cos(3*alpha))) / (1.-np.cos(alpha)**3)
	
	theta_cor = np.arctan(np.tan(theta)/B)
	theta_cor = rad2deg * theta_cor
	
	
	if lat_rot > 0 :
		lat_rot_cor = 90. - theta_cor
	else :
		lat_rot_cor = theta_cor - 90.
	
	dipole_norm_rot_cor = dir2vec(lon_rot,lat_rot_cor)
	
	
	dipole_norm_cor = np.matmul(rotmat_inv,dipole_norm_rot_cor)
	
	return dipole_norm_cor





def timeit(function, *args, **kwargs) :
	
	"""
	This function will be used as decorator to time the duration of a function
	"""
	
	def timed(*args,**kwargs) :
		t1 = time()
		result = function(*args,**kwargs)
		t2 = time()

		print('This took {:0.4f} seconds'.format(t2-t1),flush=True)
		
		return result
	
	return timed


def getIsotropicDistributionVectors(N = 1000, seed=123) :
	
	"""
	Returns a sample of N vectors drawn from an isotropic distribution
	"""
	
	N = int(N)
	
	np.random.seed(seed)

	num1 = np.random.randn(N)
	num2 = np.random.randn(N)
	num3 = np.random.randn(N)

	norm = np.sqrt(num1**2+num2**2+num3**2)

	x = num1/norm
	y = num2/norm
	z = num3/norm
	
	vectors = np.vstack((x,y,z))
	
	return vectors


def maskVectors(vec, rot_mat_list, rot_mat_inv_list, angles, galcut=30.,masking='symmetric') :
	
	"""
	This is a weird function.  It applies a mask made up of a Galactic plane mask and point source masks to a sample of vectors.
	
	The rotation matrices move those vectors which are at the center of the point source masks to the North pole.
	They are defined via getRotationMatrix_Mask.
	"""
	
	if galcut :
		
		z = vec[2,:]
		
		notmasked = np.where(np.abs(z) > np.sin(galcut * deg2rad))[0]
		vec = vec[:,notmasked]
	
	for i in range(len(angles)) :
		
		vec_rot = rotateVectors(vec, rot_mat_list[i])
		
		z = vec_rot[2,:]
		
		if masking == 'symmetric' :
			notmasked = np.where(np.abs(z) < np.cos(angles[i] * deg2rad))[0]
		elif masking == 'onesided' :
			notmasked = np.where(z < np.cos(angles[i] * deg2rad))[0]
		vec_rot = vec_rot[:,notmasked]
		
		vec = rotateVectors(vec_rot, rot_mat_inv_list[i])
	
	return vec


def getRotationMatrix_Mask(lon_list, lat_list) :
	
	"""
	Returns a list of rotation matrices and their inverse, each of which rotates vectors pointing
	towards (lonlist,latlist) to the North Galactic pole
	"""
	
	rot_mat_list = []
	rot_mat_inv_list = []
	for i in range(len(lon_list)) :
		
		rot_mat_list.append(getRotationMatrix(lon_list[i],lat_list[i]))
		rot_mat_inv_list.append(np.linalg.inv(rot_mat_list[i]))
	
	return rot_mat_list,rot_mat_inv_list


def aberrateVectors(vec, rot_mat, rot_mat_inv, vel=velocity_CMBframe) :
	
	"""
	Aberrates a sample of vectors according to a specific speed.  
	
	The direction of the velocity is part of the rotation matrix and its invers.
	
	Returns also the angular distance theta of each vector from the velocity direction.
	"""
	
	vec_rot = rotateVectors(vec, rot_mat)
	
	theta = np.arccos(vec_rot[2])
	theta *= rad2deg
	
	beta = vel / sc.c

	ct = vec_rot[2]
	st = np.sqrt(1-ct**2)
	
	theta_aberrated = np.arctan2(st*np.sqrt(1-beta**2), beta+ct)

	vec_rot[2] = np.cos(theta_aberrated)
	
	st_prime = np.sin(theta_aberrated)
	
	vec_rot[:2] *= st_prime/st
	
	vec = rotateVectors(vec_rot, rot_mat_inv)
	
	return vec,theta


def getRotationMatrix(lon=lon_CMBdipole, lat=lat_CMBdipole) :
	
	a1 = -lon
	a1 *= deg2rad
	a2 = lat-90.
	a2 *= deg2rad
	
	c1 = np.cos(a1)
	s1 = np.sin(a1)
	
	c2 = np.cos(a2)
	s2 = np.sin(a2)
	
	rot_mat = np.array([[c2*c1, -c2*s1, s2], [s1, c1, 0.], [-s2*c1, s2*s1, c2]])
	
	return rot_mat


def resampleValues(N,values,seed=123) :
	
	"""
	Returns a sample of size N drawn from the list 'values' with replacement
	"""
	
	N = int(N)
	
	np.random.seed(seed)
	sample = np.random.choice(values,size=N,replace=True)
	
	return sample


def modulateFluxes(theta, flux, alpha, vel=velocity_CMBframe) :
	
	"""
	Modulates flux values according to a velocity using the angular distance theta of each vector from the velocity direction"""
	
	theta *= deg2rad
	
	beta = vel / sc.c
	gamma = 1./np.sqrt(1-beta**2)
	
	factor = (gamma * (1+beta*np.cos(theta)))**(1+alpha)
	
	flux_mod = flux * factor
	
	return flux_mod


def cutSampleVectors(vec, flux, flux_cut) :
	
	"""
	Removes vectors whose flux falls below the flux limit
	"""
	
	indices_cut = np.where(flux > flux_cut)[0]
	
	flux_cut = flux[indices_cut]
	
	vec_cut = vec[:,indices_cut]
	
	return vec_cut,flux_cut

def sampleFlux(N,flux_amp,x,seed=123) :
    
    N = int(N)
    
    np.random.seed(seed)
    sample = np.random.uniform(size=N)
    
    flux_sample = flux_amp * (1-sample)**(-1./x)
    
    return flux_sample

def doAll_Vectors_resampling(N, N_output, rot_mat, rot_mat_inv, rot_mat_mask, rot_mat_mask_inv, maskangles, alpha, flux, x=1., galcut=30., seed=123, vel=velocity_CMBframe, do_resampling=True,W1_fluxcut=W1_fluxcut, add_isotropic=0.,estimator='linear',nside=32,masking='symmetric', mask=[None], weights=[None]) :
	
	"""
	Computes the preferred direction of a simulated sample, as well as its corrected direction, and uncorrected amplitude.
	
	Could perhaps be made a bit smarter...
	"""
	
	vec = getIsotropicDistributionVectors(N=N, seed=seed)
	vec,_ = aberrateVectors(vec, rot_mat, rot_mat_inv, vel=vel)
	vec = maskVectors(vec, rot_mat_mask, rot_mat_mask_inv, maskangles, galcut=galcut, masking=masking)
	
	vec_rot = rotateVectors(vec, rot_mat)
	theta = np.arccos(vec_rot[2]) * rad2deg
	
	N = len(vec[0,:])
	
	if do_resampling :
		flux_sample = resampleValues(N,flux,seed=seed)
		alpha_sample = resampleValues(N,alpha,seed=seed)
	else :
		flux_sample = sampleFlux(N,0.8*W1_fluxcut,x,seed=seed)
		alpha_sample = np.ones(N)*np.mean(alpha)
	
	flux_sample_mod = modulateFluxes(theta, flux_sample, alpha_sample, vel=vel)
	
	vec,_ = cutSampleVectors(vec, flux_sample_mod, W1_fluxcut)
	length = np.shape(vec)[1]
	
	if length >= N_output :
		vec = vec[:,:N_output]
	else :
		print('Vector only has length ',length,'. Pick a larger N')
		return None,None,None,None,None
	
	if add_isotropic :
		N_iso = int(add_isotropic*N_output)
		vec[:,N_output-N_iso:] = getIsotropicDistributionVectors(N=N_iso, seed=seed+123)
	
	
	length = np.shape(vec)[1]
	
	if estimator=='linear' :
		vec_dipole, d = getDipoleVectors_Crawford(vec)
		vec_dipole_corrected = correctDirection(vec_dipole,galcut=galcut)
	
		return vec,vec_dipole,vec_dipole_corrected,d,length
	
	elif estimator=='quadratic' :
		densitymap = scattomap(*vec2dir(vec),nside=nside)
		if weights[0]!=None:
			densitymap*=weights
		vec_dipole, d = getDipoleVectors_quadratic(densitymap,nside=nside,mask=mask)
		
		return vec,vec_dipole,None,d,None
	
	elif estimator=='healpy' :
		densitymap = scattomap(*vec2dir(vec),nside=nside)
		vec_dipole, d = getDipoleVectors_healpy(densitymap,mask=mask,galcut=galcut)
		return vec_dipole,None,d,None


#@timeit
def doAll_Vectors_Sim_resampling(N, N_output, simnum, alpha, flux, x=1., lon_psmask=[lon_LMC,lon_SMC], lat_psmask=[lat_LMC,lat_SMC], rad_mask=[rad_LMC,rad_SMC], seed=123, lon_direction=lon_CMBdipole, lat_direction=lat_CMBdipole, vel=velocity_CMBframe, galcut=30., do_resampling=True,W1_fluxcut=W1_fluxcut, add_isotropic=0.,estimator='linear',nside=32,masking='symmetric', mask=[None], weights=[None]) :
	
	"""
	Same as 'doAll_Vectors_Sim_resampling' but run 'simnum' times
	
	Returns longitudes and latitudes of dipole directions (uncorrected and corrected), 
	as well as dipole amplitudes (uncorrected),
	and number of remaining sources,
	for each simulation
	"""
	
	vecs_dipole = np.zeros((3,simnum))
	vecs_dipole_corrected = np.zeros((3,simnum))
	d = np.zeros(simnum)
	N_remaining = np.zeros(simnum)
	
	rot_mat = getRotationMatrix(lon=lon_direction,lat=lat_direction)
	rot_mat_inv = np.linalg.inv(rot_mat)
	
	rot_mat_mask, rot_mat_mask_inv = getRotationMatrix_Mask(lon_psmask,lat_psmask)
	
	np.random.seed(seed)
	seeds = np.random.rand(simnum)*1e9
	
	for i in range(simnum) :
		vecs_dipole[:,i], vecs_dipole_corrected[:,i], d[i], N_remaining[i] = doAll_Vectors_resampling(N, N_output, rot_mat, rot_mat_inv, rot_mat_mask, rot_mat_mask_inv, rad_mask, alpha, flux, x=x, seed=int(seeds[i]), vel=vel, galcut=galcut, do_resampling=do_resampling,W1_fluxcut=W1_fluxcut, add_isotropic=add_isotropic,estimator=estimator,nside=nside,masking=masking,mask=mask,weights=weights)
		print(str(i+1)+'/'+str(simnum),end='\r')
	
	lonlats = vec2dir(vecs_dipole)
	lonlats_corrected = vec2dir(vecs_dipole_corrected)
	
	
	return lonlats,lonlats_corrected,d,N_remaining


