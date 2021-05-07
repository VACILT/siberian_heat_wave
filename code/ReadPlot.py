import numpy as np
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from pathlib import Path
import numpy as np
from os import listdir
from os.path import isfile, join
import netCDF4
import matplotlib as mpl
import metpy as mp
from metpy.units import units
import cartopy.crs as ccrs
def preprocess(da):
    da = da.set_coords(['lev','lon','lat'])
    da = da.rename({'levs': 'lev', 'lats': 'lat', 'lons': 'lon'})
    da['lev'].attrs['long_name'] = 'altitude'
    da['lat'].attrs['long_name'] = 'latitude'
    da['lon'].attrs['long_name'] = 'longitude'
    
    return da
def ComputeAnnularMode(lat, pres, data, choice='z', hemi='infer', detrend='constant', eof_in=None, pc_in=None, eof_out=False, pc_out=False):
	"""Compute annular mode as in Gerber et al, GRL 2008.
		This is basically the first PC, but normalized to unit variance and zero mean.
		To conform to Gerber et al (2008), `data` should be anomalous height or zonal wind
		 with respect to 30-day smoothed day of year climatology.

		INPUTS:
			lat    - latitude
			pres   - pressure
			data   - variable to compute EOF from. This is typically
						geopotential or zonal wind.
						Size time x pres x lat (ie zonal mean)
			choice - not essential, but used for sign convention.
						If 'z', the sign is determined based on 70-80N/S.
						Otherwise, 50-60N/S is used.
			hemi   - hemisphere to consider
						'infer' - if mean(lat)>=0 -> NH, else SH
						'SH' or 'NH'
			detrend- detrend method for computing EOFs:
						'linear' -> remove linear trend
						'constant' -> remove total time mean
			eof_in - if None, compute EOF1 as usual.
					 if the EOF1 is already known, use this instead of
				    computing it again.
			pc_in  - if None, standardize PC1 to its own mean and std deviation
				 else, use pc_in mean and std deviation to standardize.
			eof_out- whether or not to pass the first EOF as output [False].
			pc_out - whether or not to pass the first PC as output [False].
		OUTPUT:
			AM     - The annular mode, size time x pres
			EOF    - The first EOF (if eof_out is True), size pres x lat
			PC     - The first PC (if pc_out is True). size time x pres
	"""
	#
	AM = np.full((data.shape[0],data.shape[1]),np.nan)
	if pc_out:
		pco = np.full(AM.shape,np.nan)
	# guess the hemisphere
	if hemi == 'infer':
		if np.mean(lat) >= 0:
			sgn = 1.
		else:
			sgn = -1.
	elif hemi == 'SH':
		sgn = -1.
	elif hemi == 'NH':
		sgn = 1.
	j_tmp = np.where(sgn*lat > 20)[0]
	if eof_out:
		eofo = np.full((data.shape[1],len(j_tmp)),np.nan)
	coslat = np.cos(np.deg2rad(lat))
	negCos = (coslat < 0.)
	coslat[negCos] = 0.
	# weighting as in Gerber et al GRL 2008
	sqrtcoslat = np.sqrt(coslat[j_tmp])
	# try to get the sign right
	# first possibility
	if choice == 'z':
		minj = min(sgn*70,sgn*80)
		maxj = max(sgn*80,sgn*70)
		sig = -1
	else:
		minj = min(sgn*50,sgn*60)
		maxj = max(sgn*60,sgn*50)
		sig = 1
	jj = (lat[j_tmp] > minj)*(lat[j_tmp] < maxj)
	# second possibility
	#jj = abs(lat[j_tmp]-80).argmin()
	#sig = -1
	if isinstance(pres,(int,float)):
		data = np.reshape(data,(data.shape[0],1,data.shape[1]))
		pres = [pres]
	for k in range(len(pres)):
		# remove global mean
		globZ = GlobalAvg(lat,data[:,k,:],axis=-1,lim=lat[j_tmp[0]],mx=lat[j_tmp[-1]])
		var = data[:,k,:] - globZ[:,np.newaxis]
		# area weighting: EOFs are ~variance, thus take sqrt(cos)
		var = var[:,j_tmp]*sqrtcoslat[np.newaxis,:]
		varNan = np.isnan(var)
		if np.sum(np.reshape(varNan,(np.size(varNan),)))==0:
			if eof_in is None:
				eof1,pc1,E,u,s,v = eof(var,n=1,detrend=detrend)
			else:
				pc1 = eof(var,n=1,detrend=detrend,eof_in=eof_in[k,:])
				eof1 = eof_in[k,:]
			# force the sign of PC
			pc1  = pc1*sig*np.sign(eof1[jj].mean())
			if eof_out:
				eofo[k,:] = np.squeeze(eof1)
			if pc_out:
				pco[:,k] = pc1
			# force unit variance and zero mean
			if pc_in is None:
				AM[:,k] = (pc1-pc1.mean())/np.std(pc1)
			else:
				AM[:,k] = (pc1-pc_in.mean())/np.std(pc_in)
	if eof_out and pc_out:
		return AM,eofo,pco
	elif eof_out:
		return AM,eofo
	elif pc_out:
		return AM,pco
	else:
		return AM
		
def AxRoll(x,ax,invert=False):
	"""Re-arrange array x so that axis 'ax' is first dimension.
		Undo this if invert=True
	"""
	if ax < 0:
		n = len(x.shape) + ax
	else:
		n = ax
	#
	if invert is False:
		y = np.rollaxis(x,n,0)
	else:
		y = np.rollaxis(x,0,n+1)
	return y
		
def GlobalAvg(lat,data,axis=-1,lim=20,mx=90,cosp=1):
	"""Compute cosine weighted meridional average from lim to mx.

	INPUTS:
	  lat  - latitude
	  data - data to average N x latitude
	  axis - axis designating latitude
	  lim  - starting latitude to average
	  mx   - stopping latitude to average
	  cosp - power of cosine weighting
	OUTPUTS:
	  integ- averaged data, length N
	"""
	#make sure there are more than one grid points
	if len(lat) < 2:
		return np.mean(data,axis=axis)
	#get data into the correct shape
	tmp = AxRoll(data,axis)
	shpe= tmp.shape
	tmp = np.reshape(tmp,(shpe[0],np.prod(shpe[1:])))
	#cosine weighting
	J = np.where((lat>=lim)*(lat<=mx))[0]
	coslat = np.cos(np.deg2rad(lat))**cosp
	coswgt = np.trapz(coslat[J],lat[J])
	tmp = np.trapz(tmp[J,:]*coslat[J][:,np.newaxis],lat[J],axis=0)/coswgt
	integ = np.reshape(tmp,shpe[1:])
	return integ
def ComputeEPfluxDiv(lat,pres,u,v,t,w=None,do_ubar=False,wave=-1):
	""" Compute the EP-flux vectors and divergence terms.

		The vectors are normalized to be plotted in cartesian (linear)
		coordinates, i.e. do not include the geometric factor a*cos\phi.
		Thus, ep1 is in [m2/s2], and ep2 in [hPa*m/s2].
		The divergence is in units of m/s/day, and therefore represents
		the deceleration of the zonal wind. This is actually the quantity
		1/(acos\phi)*div(F).

	INPUTS:
	  lat  - latitude [degrees]
	  pres - pressure [hPa]
	  u    - zonal wind, shape(time,p,lat,lon) [m/s]
	  v    - meridional wind, shape(time,p,lat,lon) [m/s]
	  t    - temperature, shape(time,p,lat,lon) [K]
	  w    - pressure velocity, optional, shape(time,p,lat,lon) [hPa/s]
	  do_ubar - compute shear and vorticity correction? optional
	  wave - only include this wave number. all if <0, sum over waves if a list. optional
	OUTPUTS:
	  ep1  - meridional EP-flux component, scaled to plot in cartesian [m2/s2]
	  ep2  - vertical   EP-flux component, scaled to plot in cartesian [hPa*m/s2]
	  div1 - horizontal EP-flux divergence, divided by acos\phi [m/s/d]
	  div2 - horizontal EP-flux divergence , divided by acos\phi [m/s/d]
	"""
	# some constants
	Rd    = 287.04
	cp    = 1004
	kappa = Rd/cp
	p0    = 1000
	Omega = 2*np.pi/(24*3600.) # [1/s]
	a0    = 6.371e6
	# geometry
	pilat = lat*np.pi/180
	dphi  = np.gradient(pilat)[np.newaxis,np.newaxis,:]
	coslat= np.cos(pilat)[np.newaxis,np.newaxis,:]
	sinlat= np.sin(pilat)[np.newaxis,np.newaxis,:]
	R     = 1./(a0*coslat)
	f     = 2*Omega*sinlat
	pp0  = (p0/pres[np.newaxis,:,np.newaxis])**kappa
	dp    = np.gradient(pres)[np.newaxis,:,np.newaxis]
	#
	# absolute vorticity
	if do_ubar:
		ubar = np.nanmean(u,axis=-1)
		fhat = R*np.gradient(ubar*coslat,edge_order=2)[-1]/dphi
	else:
		fhat = 0.
	fhat = f - fhat # [1/s]
	#
	## compute thickness weighted heat flux [m.hPa/s]
	vbar,vertEddy = ComputeVertEddy(v,t,pres,p0,wave) # vertEddy = bar(v'Th'/(dTh_bar/dp))
	#
	## get zonal anomalies
	u = GetAnomaly(u)
	v = GetAnomaly(v)
	if isinstance(wave,list):
		upvp = np.sum(GetWaves(u,v,wave=-1)[:,:,:,wave],-1)
	elif wave<0:
		upvp = np.nanmean(u*v,axis=-1)
	else:
		upvp = GetWaves(u,v,wave=wave)
	#
	## compute the horizontal component
	if do_ubar:
		shear = np.gradient(ubar,edge_order=2)[1]/dp # [m/s.hPa]
	else:
		shear = 0.
	ep1_cart = -upvp + shear*vertEddy # [m2/s2 + m/s.hPa*m.hPa/s] = [m2/s2]
	#
	## compute vertical component of EP flux.
	## at first, keep it in Cartesian coordinates, ie ep2_cart = f [v'theta'] / [theta]_p + ...
	#
	ep2_cart = fhat*vertEddy # [1/s*m.hPa/s] = [m.hPa/s2]
	if w is not None:
		w = GetAnomaly(w) # w = w' [hPa/s]
		if isinstance(wave,list):
			w = sum(GetWaves(u,w,wave=wave)[:,:,:,wave],-1)
		elif wave<0:
			w = np.nanmean(w*u,axis=-1) # w = bar(u'w') [m.hPa/s2]
		else:
			w = GetWaves(u,w,wave=wave) # w = bar(u'w') [m.hPa/s2]
		ep2_cart = ep2_cart - w # [m.hPa/s2]
	#
	#
	# We now have to make sure we get the geometric terms right
	# With our definition,
	#  div1 = 1/(a.cosphi)*d/dphi[a*cosphi*ep1_cart*cosphi],
	#    where a*cosphi comes from using cartesian, and cosphi from the derivative
	# With some algebra, we get
	#  div1 = cosphi d/d phi[ep1_cart] - 2 sinphi*ep1_cart
	div1 = coslat*np.gradient(ep1_cart,edge_order=2)[-1]/dphi - 2*sinlat*ep1_cart
	# Now, we want acceleration, which is div(F)/a.cosphi [m/s2]
	div1 = R*div1 # [m/s2]
	#
	# Similarly, we want acceleration = 1/a.coshpi*a.cosphi*d/dp[ep2_cart] [m/s2]
	div2 = np.gradient(ep2_cart,edge_order=2)[1]/dp # [m/s2]
	#
	# convert to m/s/day
	div1 = div1*86400
	div2 = div2*86400
	#
	return ep1_cart,ep2_cart,div1,div2
	
def eof(X,n=-1,detrend='constant',eof_in=None):
	"""Principal Component Analysis / Empirical Orthogonal Functions / SVD

		Uses Singular Value Decomposition to find the dominant modes of variability.
		The field X can be reconstructed with Y = dot(EOF,PC) + X.mean(axis=time)

		INPUTS:
			X	-- Field, shape (time x space).
			n	-- Number of modes to extract. All modes if n < 0
			detrend -- detrend with global mean ('constant')
						  or linear trend ('linear')
		    eof_in  -- If not None, compute PC by projecting eof onto X.
		OUTPUTS:
			EOF - Spatial modes of variability
			PC  - Temporal evolution of EOFs - only output if eof_in is not None
			E   - Explained value of variability
			u   - spatial modes
			s   - variances
			v   - temporal modes
	"""
	import scipy.signal as sg
	# make sure we have a matrix time x space
	shpe = X.shape
	if len(shpe) > 2:
		X = X.reshape([shpe[0],np.prod(shpe[1:])])
		if eof_in is not None:
			eof_in = eof_in.reshape([np.prod(eof_in.shape[:-1]),eof_in.shape[-1]])
	# take out the time mean or trend
	X = sg.detrend(X.transpose(),type=detrend)
	if eof_in is not None:
		if eof_in.shape[-1] == X.shape[0]:
			PC =  np.matmul(eof_in, X)
			eof_norm = np.dot(eof_in.transpose(),eof_in)
			return np.dot(PC,np.linalg.inv(eof_norm))
		else:
			PC = np.matmul(eof_in.transpose(), X)
			eof_norm = np.dot(eof_in.transpose(),eof_in)
			return np.dot(PC.transpose(),np.linalg.inv(eof_norm)).transpose()
		# return sg.detrend(PC,type='constant')
	# perform SVD - v is actually V.H in X = U*S*V.H
	u,s,v = np.linalg.svd(X, full_matrices=False)
	# now, u contains the spatial, and v the temporal structures
	# s contains the variances, with the same units as the input X
	# u.shape = (space, modes(space)), v.shape = (modes(space), time)

	# get the first n modes, in physical units
	#  we can either project the data onto the principal component, X*V
	#  or multiply u*s. This is the same, as U*S*V.H*V = U*S
	if n < 0:
		n = s.shape[0]
	EOF = np.dot(u[:,:n],np.diag(s)[:n,:n])
	# time evolution is in v
	PC  = v[:n,:]
	# EOF wants \lambda = the squares of the eigenvalues,
	#  but SVD yields \gamma = \sqrt{\lambda}
	s2 = s*s
	E   = s2[:n]/sum(s2)
	# now we need to make sure we get everything into the correct shape again
	u = u[:,:n]
	s = s[:n]
	v = v.transpose()[:,:n]
	if len(shpe) > 2:
		# replace time dimension with modes at the end of the array
		newshape = list(shpe[1:])+[n]
		EOF = EOF.reshape(newshape)
		u   = u	 .reshape(newshape)
	return EOF,PC,E,u,s,v
	
	
Months=['Jan', 'Feb', 'Mar']

###2020
root_path_1 = Path('/home/gemeinsam_tmp/VACILT/Sibirian_Heatwave/Sib_Heat_Out/20201219_LH_version_10km_assimilation/nc/')
in_folders_2020 = [f for f in listdir(root_path_1) if isfile(join(root_path_1, f))]

###2008-2018
root_path_ref = Path('/home/gemeinsam_tmp/VACILT/Sibirian_Heatwave/Sib_Heat_Ref_Out/20210104_LH_version_10km_assimilation/nc/')
in_folders_ref = [f for f in listdir(root_path_ref) if isfile(join(root_path_ref, f))]


###all filepaths of 2020
filepath_2020=[]
for i in range(0,len(in_folders_2020)):
	fp_2020 = str(root_path_1)+'/'+str(in_folders_2020[i])
	filepath_2020.append(fp_2020)

###all filepaths 2008-2018
filepath_ref=[]
for i in range(0,len(in_folders_ref)):
	fp_ref = str(root_path_ref)+'/'+str(in_folders_ref[i])
	filepath_ref.append(fp_ref)

###create datasets the most unprofessional way possible
ds_2020_Jan = xr.open_dataset(filepath_2020[0])
ds_ref_Jan = xr.open_dataset(filepath_ref[0])
ds_2020_Feb = xr.open_dataset(filepath_2020[1])
ds_ref_Feb = xr.open_dataset(filepath_ref[1])
ds_2020_Mar = xr.open_dataset(filepath_2020[2])
ds_ref_Mar = xr.open_dataset(filepath_ref[2])

###preprocess datasets the most unprofessional way possible
ds_2020_Jan=preprocess(ds_2020_Jan)
ds_ref_Jan=preprocess(ds_ref_Jan)
ds_2020_Feb=preprocess(ds_2020_Feb)
ds_ref_Feb=preprocess(ds_ref_Feb)
ds_2020_Mar=preprocess(ds_2020_Mar)
ds_ref_Mar=preprocess(ds_ref_Mar)

def heighttopress(height,temp):
	height=height*1000
	press=np.exp(-height/7000)*1000 
	#press = (np.float128(1013.25*np.exp(-(9.81*height*0.02896968)/(288.16*8.314462618))))*100
	return press
	
	

meantemp=ds_2020_Jan['tem'].sel(lat = 30).mean(['time','lon']) 


ds_2020_Jan['lev']= heighttopress(ds_2020_Jan['lev'], meantemp) 
ds_ref_Jan['lev']=heighttopress(ds_ref_Jan['lev'], meantemp) 
ds_2020_Feb['lev']=heighttopress(ds_2020_Feb['lev'], meantemp) 
ds_ref_Feb['lev']=heighttopress(ds_ref_Feb['lev'], meantemp) 
ds_2020_Mar['lev']=heighttopress(ds_2020_Mar['lev'], meantemp) 
ds_ref_Mar['lev']=heighttopress(ds_ref_Mar['lev'], meantemp) 




#AO
AO_Jan = ComputeAnnularMode(ds_2020_Jan['lat'].values, ds_2020_Jan['lev'].values, ds_2020_Jan['phi'].mean('lon').values ) 
AO_ref_Jan=ComputeAnnularMode(ds_ref_Jan['lat'].values, ds_ref_Jan['lev'].values, ds_ref_Jan['phi'].mean('lon').values )
AO_Feb=ComputeAnnularMode(ds_2020_Feb['lat'].values, ds_2020_Feb['lev'].values, ds_2020_Feb['phi'].mean('lon').values )
AO_ref_Feb=ComputeAnnularMode(ds_ref_Feb['lat'].values, ds_ref_Feb['lev'].values, ds_ref_Feb['phi'].mean('lon').values )
AO_Mar=ComputeAnnularMode(ds_2020_Mar['lat'].values, ds_2020_Mar['lev'].values, ds_2020_Mar['phi'].mean('lon').values )
AO_ref_Mar=ComputeAnnularMode(ds_ref_Mar['lat'].values, ds_ref_Mar['lev'].values, ds_ref_Mar['phi'].mean('lon').values )

AO_Jan_Arr=np.round(xr.DataArray(AO_Jan, dims=['time','lev']),2)
AO_Jan_Ref_Arr=np.round(xr.DataArray(AO_ref_Jan, dims=['time','lev']),2)
AO_Feb_Arr=np.round(xr.DataArray(AO_Feb, dims=['time','lev']),2)
AO_Feb_Ref_Arr=np.round(xr.DataArray(AO_ref_Feb, dims=['time','lev']),2)
AO_Mar_Arr=np.round(xr.DataArray(AO_Mar, dims=['time','lev']),2)
AO_Mar_Ref_Arr=np.round(xr.DataArray(AO_ref_Mar, dims=['time','lev']),2)

AO_Jan_Arr['lev']=ds_2020_Jan['lev']
AO_Feb_Arr['lev']=ds_2020_Jan['lev']
AO_Mar_Arr['lev']=ds_2020_Jan['lev']




#diff_jan= AO_Jan_Arr-AO_Jan_Ref_Arr
#diff_feb= mean_zon_Feb-mean_zon_Feb_ref
#diff_mar= mean_zon_Mar-mean_zon_Mar_ref

mean_AO_Jan=AO_Jan_Arr.mean('time')#.sel(lat = 30).mean('lon')
#mean_zon_Jan_ref=ds_ref_Jan['zon'].sel(lat = 30).mean(['time','lon'])#.sel(lat = 30).mean('lon')
mean_AO_Feb=AO_Feb_Arr.mean('time')#.sel(lat = 30).mean('lon')
#mean_zon_Feb_ref=ds_ref_Feb['zon'].sel(lat = 30).mean(['time','lon'])#.sel(lat = 30).mean('lon')
mean_AO_Mar=AO_Mar_Arr.mean('time')#.sel(lat = 30).mean('lon')
#mean_zon_Mar_ref=ds_ref_Mar['zon'].sel(lat = 30).mean(['time','lon'])#.sel(lat = 30).mean('lon')

#AO_all = xr.concat([AO_Jan_Arr, AO_Feb_Arr, AO_Mar_Arr], dim = 'month')
#AO_all['month'] = [1, 2, 3]

#meanzon
mean_zon_Jan=ds_2020_Jan['zon'].sel(lat = 30).mean(['time','lon'])#.sel(lat = 30).mean('lon')
mean_zon_Jan_ref=ds_ref_Jan['zon'].sel(lat = 30).mean(['time','lon'])#.sel(lat = 30).mean('lon')
mean_zon_Feb=ds_2020_Feb['zon'].sel(lat = 30).mean(['time','lon'])#.sel(lat = 30).mean('lon')
mean_zon_Feb_ref=ds_ref_Feb['zon'].sel(lat = 30).mean(['time','lon'])#.sel(lat = 30).mean('lon')
mean_zon_Mar=ds_2020_Mar['zon'].sel(lat = 30).mean(['time','lon'])#.sel(lat = 30).mean('lon')
mean_zon_Mar_ref=ds_ref_Mar['zon'].sel(lat = 30).mean(['time','lon'])#.sel(lat = 30).mean('lon')

mothticks=[ 1, 2 , 3]
clim=xr.concat([mean_zon_Jan_ref, mean_zon_Feb_ref, mean_zon_Mar_ref], dim = 'month')
clim2020=xr.concat([mean_zon_Jan, mean_zon_Feb, mean_zon_Mar], dim = 'month')

diff_jan= mean_zon_Jan-mean_zon_Jan_ref
diff_feb= mean_zon_Feb-mean_zon_Feb_ref
diff_mar= mean_zon_Mar-mean_zon_Mar_ref

da_all = xr.concat([diff_jan, diff_feb, diff_mar], dim = 'month')
da_all['month'] = [1, 2, 3]
da_all['lev'].attrs['units'] = "hPa"
da_all.attrs['units'] = "m/s"



#meantemp
mean_temp_Jan=ds_2020_Jan['zon'].sel(lev =  8.161e+02, method='nearest').mean('time')#.sel(lat = 30).mean('lon')
mean_temp_Jan_ref=ds_ref_Jan['zon'].sel(lev =  8.161e+02, method='nearest').mean('time')#.sel(lat = 30).mean('lon')
mean_temp_Feb=ds_2020_Feb['zon'].sel(lev =  8.161e+02, method='nearest').mean('time')#.sel(lat = 30).mean('lon')
mean_temp_Feb_ref=ds_ref_Feb['zon'].sel(lev =  8.161e+02, method='nearest').mean('time')#.sel(lat = 30).mean('lon')
mean_temp_Mar=ds_2020_Mar['zon'].sel(lev =  8.161e+02, method='nearest').mean('time')#.sel(lat = 30).mean('lon')
mean_temp_Mar_ref=ds_ref_Mar['zon'].sel(lev =  8.161e+02, method='nearest').mean('time')#.sel(lat = 30).mean('lon')

mean_temp2_Jan=ds_2020_Jan['tem'].sel(lev =  8.161e+02, method='nearest').mean('time')#.sel(lat = 30).mean('lon')
mean_temp2_Jan_ref=ds_ref_Jan['tem'].sel(lev =  8.161e+02, method='nearest').mean('time')#.sel(lat = 30).mean('lon')
mean_temp2_Feb=ds_2020_Feb['tem'].sel(lev =  8.161e+02, method='nearest').mean('time')#.sel(lat = 30).mean('lon')
mean_temp2_Feb_ref=ds_ref_Feb['tem'].sel(lev =  8.161e+02, method='nearest').mean('time')#.sel(lat = 30).mean('lon')
mean_temp2_Mar=ds_2020_Mar['tem'].sel(lev =  8.161e+02, method='nearest').mean('time')#.sel(lat = 30).mean('lon')
mean_temp2_Mar_ref=ds_ref_Mar['tem'].sel(lev =  8.161e+02, method='nearest').mean('time')#.sel(lat = 30).mean('lon')

diff_jan_zon_glob= mean_temp_Jan-mean_temp_Jan_ref
diff_feb_zon_glob= mean_temp_Feb-mean_temp_Feb_ref
diff_mar_zon_glob= mean_temp_Mar-mean_temp_Mar_ref

diff_jan_mer_glob= mean_temp2_Jan-mean_temp2_Jan_ref
diff_feb_mer_glob= mean_temp2_Feb-mean_temp2_Feb_ref
diff_mar_mer_glob= mean_temp2_Mar-mean_temp2_Mar_ref

#test=diff_jan_mer_glob+diff_jan_zon_glob



# meanAO_Jan=AO_Jan_Arr.mean('time')
# meanAO_Feb=AO_Feb_Arr.mean('time')
# meanAO_Mar=AO_Mar_Arr.mean('time')

# AO_all = xr.concat([meanAO_Jan, meanAO_Feb, meanAO_Mar], dim = 'month')
# AO_all['month'] = [1, 2, 3]
# AO_all['lev']=ds_2020_Jan['lev']
# print(AO_all['lev'])
# y1=AO_Jan_Arr.mean('time')
# y2=AO_Jan_Ref_Arr
# #y1['time']= y1['time']/20
# #y2['time']=y2['time']/20
#ticks=[100,10,1,0.1]


# y1=AO_Feb_Arr

# y1['time']= y1['time']/20

# plt.figure(figsize=(8, 6))
# title='AO_Feb_2020'
# y1.plot.contourf(x="time", levels=21, robust=True, yincrease=False)
# #plt.ylim(816, 320)
# #plt.yticks(ticks)
# plt.ylim(816,21)
# plt.ylabel('lev[hPa]')
# plt.legend()
# plt.xlim(0,31)
# plt.title(title)
# #plt.xticks(mothticks)
# #plt.yscale('log')
# plt.savefig('/home/akoetsche/git/Plots/'+title+'.pdf')


plt.figure(figsize=(8, 6))
title='Geopot_850hPa_Jan'
p=ds_2020_Jan['phi'].sel(lev=8.161e+02, method='nearest').sel(time=5).plot.contourf(subplot_kws=dict(projection=ccrs.Orthographic(60, 70), facecolor="gray"),transform=ccrs.PlateCarree(), levels=61)
plt.title(title)
p.axes.set_global()
p.axes.coastlines()
plt.savefig('/home/akoetsche/git/Plots/'+title+'.pdf')


# y1['time']= y1['time']/20
# y2['time']=y2['time']/20
# plt.figure(figsize=(8, 6))
# title='AO_Mar_2020_and_AO_Mar_2008-2018'
# y1.sel(lev=0).plot(label='AO 2020', x='time', color='k', linewidth=1)
# y2.sel(lev=0).plot(label='AO 2008-2018',x='time', color='orange')
# plt.fill_between(y1['time'], y1.sel(lev=0), y2.sel(lev=0), where= y1.sel(lev=0) >=  y2.sel(lev=0),color='blue', interpolate=True)
# plt.fill_between(y1['time'], y1.sel(lev=0), y2.sel(lev=0),where= y1.sel(lev=0) <=  y2.sel(lev=0),color='red', interpolate=True)
# plt.ylabel('AO-Index 850hPa')
# plt.xlabel('Day')
# plt.title(title)
# plt.xlim(0,31)
# plt.legend()
# plt.savefig('/home/akoetsche/git/Plots/'+title+'.pdf')




#pivot=The part of the arrow that is at the grid point; the arrow rotates about this point

# import pylab as P
# print('ja')
# plt.figure(figsize=(8, 6))
# title='850hPa_Temp_Anomalies_Jan'
# p= diff_jan_mer_glob.plot.contourf(subplot_kws=dict(projection=ccrs.Orthographic(60, 60), facecolor="gray"),transform=ccrs.PlateCarree(), levels=51)

# plt.title(title)
# p.axes.set_global()
# p.axes.coastlines()
# plt.savefig('/home/akoetsche/git/Plots/'+title+'.pdf')

# ticks=[100,10,1,0.1]

# plt.figure(figsize=(8, 6))
# clim2020.plot.contourf(x="t", levels=21, robust=True, yincrease=False)
# title='62.5°N Zonal Mean Wind'
# plt.title(title)
# #plt.yticks(ticks)
# #plt.xticks(mothticks)
# plt.yscale('log')
# plt.savefig('/home/akoetsche/git/Plots/'+title+'.pdf')
