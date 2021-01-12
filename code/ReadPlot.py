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



###2020
root_path_1 = Path('/home/gemeinsam_tmp/VACILT/Sibirian_Heatwave/Sib_Heat_Out/20201219_LH_version_10km_assimilation/nc/')
in_folders_2020 = [f for f in listdir(root_path_1) if isfile(join(root_path_1, f))]

###2008-2018
root_path_ref = Path('/home/gemeinsam_tmp/VACILT/Sibirian_Heatwave/Sib_Heat_Ref_Out/20210104_LH_version_10km_assimilation/nc/')
in_folders_ref = [f for f in listdir(root_path_ref) if isfile(join(root_path_ref, f))]


###path2020
filepath_2020=[]
for i in range(0,len(in_folders_2020)):
	fp_2020 = str(root_path_1)+'/'+str(in_folders_2020[i])
	filepath_2020.append(fp_2020)

###path2008-2018
filepath_ref=[]
for i in range(0,len(in_folders_ref)):
	fp_ref = str(root_path_ref)+'/'+str(in_folders_ref[i])
	filepath_ref.append(fp_ref)


def get_nc_data(thisfile, varname):
		#print('loading variable '+varname +' from ' + thisfile)
		ncfile = netCDF4.Dataset(thisfile)
		var = ncfile.variables[varname][:]
		#requested_var=np.array(var.getValue())
		requested_var=np.array(var)
		requested_var[np.isnan(requested_var)]=0.0
		ncfile.close()
		return requested_var

ds_2020 = xr.open_dataset(filepath_2020[1])
print(ds_2020)

ncfiles=filepath_2020

if ncfiles:
	file = ncfiles[0]
	zon = get_nc_data(file, 'zon')
	mer    = get_nc_data(file, 'mer')*1.
	tem        = get_nc_data(file,'tem')*1.
	phi       = get_nc_data(file,'phi')*1.
	ver       = get_nc_data(file,'ver')*1.
	day     = get_nc_data(file,'day')*1.
	lev     = get_nc_data(file,'lev')*1.	
	lat     = get_nc_data(file,'lat')*1.	
	lon     = get_nc_data(file,'lon')*1.	

	for file in ncfiles[1:]:
		zon        = np.append(zon,get_nc_data(file,'zon')*1., axis=0)
		mer       = np.append(mer,get_nc_data(file,'mer')*1., axis=0)
		tem   	  = np.append(tem,get_nc_data(file,'tem')*1., axis=0)
		phi = np.append(phi, get_nc_data(file, 'phi')) 	
		ver = np.append(ver, get_nc_data(file, 'ver')) 	
		day = np.append(day, get_nc_data(file, 'day')) 	
		lat = np.append(lat, get_nc_data(file, 'lat')) 	
		lon = np.append(lon, get_nc_data(file, 'lon')) 	
						
			
print(ComputeVertEddy(ver,tem,lev))