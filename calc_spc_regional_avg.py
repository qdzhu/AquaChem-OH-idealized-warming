import pandas as pd
import numpy as np
from scipy import signal
import numpy.polynomial.polynomial as poly
import numpy.ma as ma
from netCDF4 import Dataset
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from glob import glob
#import cartopy.crs as ccrs
from IPython.display import display
#from cartopy.util import add_cyclic_point
#import cartopy
#import cartopy.crs as ccrs
#from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy.interpolate import interp2d
#map_projection = ccrs.Mollweide
import seaborn as sns
import statsmodels.api as sm
import argparse

data = xr.open_dataset('/net/fs09/d0/qdzhu/result/Nicole_aquaQPC6_f09_AveChem_LBC_IC/run/AveChem_LBC_IC.i.nc')
lat = data['lat'].values
lon = data['lon'].values
lev = data['lev'].values
lev = lev[8:]

def calc_box_oh(spc, pres_indx, lat_indx, s_year, e_year, output_strs, avg_opt):
    #output_strs = ['FC2000','FC2000_SST_plus2']
    nyear = e_year - s_year
    
    spcs = np.zeros((len(output_strs),nyear,24,192,288))
    mass_trop = np.zeros((len(output_strs),nyear,24,192,288))
    volumes = np.zeros((len(output_strs),nyear,24,192,288))
    for i_output, output_str in enumerate(output_strs):
        for i_year, year in enumerate(range(s_year,e_year)):
            mass = np.load('../Output/camchem_{}_3d/{}_year_{}.npy'.format(output_str,'MASS_trop',year))[8:,:,:]
            mass_trop[i_output,i_year,:,:,:] = mass

            volume = np.load('../Output/camchem_{}_3d/{}_year_{}.npy'.format(output_str,'volume',year))[8:,:,:]
            volumes[i_output,i_year,:,:,:] = volume

            oh = np.load('../Output/camchem_{}_3d/{}_year_{}.npy'.format(output_str,spc,year))[8:,:,:]
            if spc == 'CH4':
                oh = oh*6.02e23/(28.9643*1e-3)*density*1e-6*volume/(6.02e23)*16*1e-12
            elif spc == 'CH4_CHML':
                oh = oh*volume/(6.02e23)*1e-12*86400*365*16
            spcs[i_output,i_year,:,:,:] = oh
    
    this_oh_box = np.zeros((len(output_strs),nyear))
    
    this_mass = mass_trop[:,:,pres_indx,:,:][:,:,:,lat_indx,:]
    this_size = int(this_mass.size/nyear/len(output_strs))
    this_mass = this_mass.reshape((len(output_strs),nyear, this_size))
    
    this_volume = volumes[:,:,pres_indx,:,:][:,:,:,lat_indx,:]
    this_volume = this_volume.reshape((len(output_strs),nyear, this_size))
    
    this_oh = spcs[:,:,pres_indx,:,:][:,:,:,lat_indx,:]
    this_oh = this_oh.reshape((len(output_strs),nyear, this_size))

    if avg_opt =='volume':
        if spc == 'CH4_CHML':
            this_oh_box = np.nansum(this_oh,axis=2)
        else:
            this_oh_box = np.nansum(this_oh * this_volume,axis=2)/np.nansum(this_volume,axis=2)
    elif avg_opt =='mass':
        this_oh_box = np.nansum(this_oh * this_mass,axis=2)/np.nansum(this_mass,axis=2)

    oh_avg = np.nanmean(this_oh_box,axis=1)
    oh_std = np.nanstd(this_oh_box,axis=1)
    return oh_avg, oh_std

def calc_box_oh_2d(spc, pres_indx, lat_indx, s_year, e_year, output_strs, avg_opt):
    nyear = e_year - s_year
    
    spcs = np.zeros((len(output_strs),nyear,192,288))
    mass_trop = np.zeros((len(output_strs),nyear,192,288))
    volumes = np.zeros((len(output_strs),nyear,192,288))
    for i_output, output_str in enumerate(output_strs):
        for i_year, year in enumerate(range(s_year,e_year)):
            oh = np.load('../Output/camchem_{}_3d/{}_year_{}.npy'.format(output_str,spc,year))[:,:]
            spcs[i_output,i_year,:,:] = oh
    
    this_oh_box = np.zeros((len(output_strs),nyear))
    
    this_oh = spcs[:,:,lat_indx,:]
    this_size = int(this_oh.size/nyear/len(output_strs))
    this_oh = this_oh.reshape((len(output_strs),nyear, this_size))
    this_oh_box = np.nanmean(this_oh,axis=2)
    oh_avg = np.nanmean(this_oh_box,axis=1)
    oh_std = np.nanstd(this_oh_box,axis=1)
    return oh_avg, oh_std

def calc_aqua_box_oh(spc, pres_indx, lat_indx, s_year, e_year, output_strs, avg_opt):
    #output_strs = ['ZonalChem_addMEG_emis','ZonalChem_addMEG_2Demis',            'ZonalChem_plus2_addMEG_plus2_emis','ZonalChem_plus2_addMEG_plus2_2Demis',
    #           'ZonalChem_plus2_addMEG_emis']
    #output_strs = ['ZonalChem_addMEG_emis']
    #nyear = 3
    nyear = e_year - s_year
    aqua_spcs = np.zeros((len(output_strs),nyear,12,24,192,288))
    aqua_mass_trop = np.zeros((len(output_strs),nyear,12,24,192,288))
    aqua_volumes = np.zeros((len(output_strs),nyear,12,24,192,288))

    for i_output, output_str in enumerate(output_strs):
        for i_year, year in enumerate(range(s_year,e_year)):
            density = np.load('../Output/aquas_{}_3d/{}_month_year_{}.npy'.format(output_str,'density',year))[:,8:,:,:]
            
            oh = np.load('../Output/aquas_{}_3d/{}_month_year_{}.npy'.format(output_str,spc,year))[:,8:,:,:]
            
            mass = np.load('../Output/aquas_{}_3d/{}_month_year_{}.npy'.format(output_str,'MASS_trop',year))[:,8:,:,:]
            aqua_mass_trop[i_output,i_year,:,:,:,:] = mass
            
            volume = np.load('../Output/aquas_{}_3d/{}_month_year_{}.npy'.format(output_str,'volume',year))[:,8:,:,:]
            aqua_volumes[i_output,i_year,:,:,:,:] = volume
            
            
            if spc == 'CH4':
                oh = oh*6.02e23/(28.9643*1e-3)*density*1e-6*volume/(6.02e23)*16*1e-12

            if spc == 'CH4_CHML':
                oh = oh*volume/(6.02e23)*1e-12*86400*365*16
            aqua_spcs[i_output,i_year,:,:,:,:] = oh

    aqua_spcs = aqua_spcs.reshape((len(output_strs),nyear*12,24,192,288))
    aqua_mass_trop = aqua_mass_trop.reshape((len(output_strs),nyear*12,24,192,288))
    aqua_volumes = aqua_volumes.reshape((len(output_strs),nyear*12,24,192,288))
    
    aqua_spcs = aqua_spcs[:,24:,:,:,:]
    aqua_mass_trop = aqua_mass_trop[:,24:,:,:,:]
    aqua_volumes = aqua_volumes[:,24:,:,:,:]
    
    this_oh_box = np.zeros((len(output_strs),aqua_spcs.shape[1]))
    
    this_mass = aqua_mass_trop[:,:,pres_indx,:,:][:,:,:,lat_indx,:]    
    this_volume = aqua_volumes[:,:,pres_indx,:,:][:,:,:,lat_indx,:]
    this_oh = aqua_spcs[:,:,pres_indx,:,:][:,:,:,lat_indx,:]

    for i_output in range(len(output_strs)):
        for i_time in range(aqua_spcs.shape[1]):
            if avg_opt == 'volume':
                if spc == 'CH4_CHML':
                    this_oh_box[i_output,i_time] = np.nansum(this_oh[i_output,i_time,:,:,:])
                    #9print(this_oh_box[i_output,i_time])
                else:
                    this_oh_box[i_output,i_time] = np.nansum(this_oh[i_output,i_time,:,:,:]*this_volume[i_output,i_time,:,:,:])/np.sum(this_volume[i_output,i_time,:,:,:])
            elif avg_opt == 'mass':
                this_oh_box[i_output,i_time] = np.nansum(this_oh[i_output,i_time,:,:,:]*this_mass[i_output,i_time,:,:,:])/np.sum(this_mass[i_output,i_time,:,:,:])

    oh_avg = np.nanmean(this_oh_box,axis=1)
    oh_std = np.nanstd(this_oh_box,axis=1)
    return oh_avg, oh_std

def calc_aqua_box_oh_2d(spc, pres_indx, lat_indx, s_year, e_year, output_strs, avg_opt):
    #output_strs = ['ZonalChem_addMEG_emis','ZonalChem_addMEG_2Demis',            'ZonalChem_plus2_addMEG_plus2_emis','ZonalChem_plus2_addMEG_plus2_2Demis',
    #           'ZonalChem_plus2_addMEG_emis']
    #output_strs = ['ZonalChem_addMEG_emis']
    #nyear = 3
    nyear = e_year - s_year
    aqua_spcs = np.zeros((len(output_strs),nyear,12,192,288))

    for i_output, output_str in enumerate(output_strs):
        for i_year, year in enumerate(range(s_year,e_year)):
            oh = np.load('../Output/aquas_{}_3d/{}_month_year_{}.npy'.format(output_str,spc,year))[:,:,:]
            aqua_spcs[i_output,i_year,:,:,:] = oh

    aqua_spcs = aqua_spcs.reshape((len(output_strs),nyear*12,192,288))
   
    aqua_spcs = aqua_spcs[:,24:,:,:]
    
    this_oh_box = np.zeros((len(output_strs),aqua_spcs.shape[1]))
    
    this_oh = aqua_spcs[:,:,lat_indx,:]

    for i_output in range(len(output_strs)):
        for i_time in range(aqua_spcs.shape[1]):
            this_oh_box[i_output,i_time] = np.nanmean(this_oh[i_output,i_time,:,:])

    oh_avg = np.nanmean(this_oh_box,axis=1)
    oh_std = np.nanstd(this_oh_box,axis=1)
    return oh_avg, oh_std

def create_df_camchem(spc, pres_indx, lat_indx, label, s_year, e_year, output_strs, avg_opt):
    df_comb = pd.DataFrame()
    oh_avg, oh_std = calc_box_oh(spc, pres_indx, lat_indx, s_year, e_year, output_strs, avg_opt)
    this_df = pd.DataFrame()
    this_df.loc[0,spc+'_avg'] = oh_avg[0]
    this_df.loc[0,spc+'_std'] = oh_std[0]
    this_df.loc[:,'model'] = 'CAM-Chem'
    this_df.loc[:,'region'] = label
    df_comb =pd.concat([df_comb, this_df])
    return df_comb

def create_df_camchem_2d(spc, pres_indx, lat_indx, label, s_year, e_year, output_strs, avg_opt):
    df_comb = pd.DataFrame()
    oh_avg, oh_std = calc_box_oh_2d(spc, pres_indx, lat_indx, s_year, e_year, output_strs, avg_opt)
    this_df = pd.DataFrame()
    
    this_df.loc[0,spc+'_avg'] = oh_avg[0]
    this_df.loc[0,spc+'_std'] = oh_std[0]
    this_df.loc[:,'model'] = 'CAM-Chem'
    this_df.loc[:,'region'] = label
    df_comb =pd.concat([df_comb, this_df])
    return df_comb

def create_df_aquachem(spc, pres_indx, lat_indx, label, model_label, s_year, e_year, output_strs, avg_opt):
    df_comb = pd.DataFrame()
    aquas_avg, aquas_std = calc_aqua_box_oh(spc, pres_indx, lat_indx, s_year, e_year, output_strs, avg_opt)
    this_df = pd.DataFrame()
    this_df.loc[0,spc+'_avg'] = aquas_avg[0]
    this_df.loc[0,spc+'_std'] = aquas_std[0]
    this_df.loc[:,'model'] = model_label
    this_df.loc[:,'region'] = label
    df_comb =pd.concat([df_comb, this_df])
    return df_comb

def create_df_aquachem_2d(spc, pres_indx, lat_indx, label, model_label, s_year, e_year, output_strs, avg_opt):
    df_comb = pd.DataFrame()
    aquas_avg, aquas_std = calc_aqua_box_oh_2d(spc, pres_indx, lat_indx, s_year, e_year, output_strs, avg_opt)
    this_df = pd.DataFrame()
    this_df.loc[0,spc+'_avg'] = aquas_avg[0]
    this_df.loc[0,spc+'_std'] = aquas_std[0]
    this_df.loc[:,'model'] = model_label
    this_df.loc[:,'region'] = label
    df_comb =pd.concat([df_comb, this_df])
    return df_comb

def create_lat_bins_aquachem_2d(spc, pres_indx,pres,model_label, s_year, e_year, output_strs, avg_opt):
    
    df_comb = pd.DataFrame()
    df_comb_abs = pd.DataFrame()
    
    lat_indx = (lat >= -20) & (lat <= 20)
    label = 'Tropical'
    
    df = create_df_aquachem_2d(spc, pres_indx, lat_indx, label, model_label, s_year, e_year, output_strs, avg_opt)
    df_comb = pd.concat([df_comb, df])

    
    lat_indx = ((lat <-20) & (lat >-60)) 
    label = 'S Extratrop'
    df = create_df_aquachem_2d(spc, pres_indx, lat_indx, label, model_label, s_year, e_year, output_strs, avg_opt)
    df_comb = pd.concat([df_comb, df])

    
    lat_indx = (lat > 20) & (lat < 60)
    label = 'N Extratrop'
    df = create_df_aquachem_2d(spc, pres_indx, lat_indx, label, model_label, s_year, e_year, output_strs, avg_opt)
    df_comb = pd.concat([df_comb, df])

    
    filepath = '/home/qdzhu/fs09/Stratocu-planet/Chemistry/Output/dfs/'
    filename = '{}_{}_pres_{}_avg_{}.csv'.format(model_label, spc,pres, avg_opt)
    df_comb.to_csv(filepath + filename)
    
    
    

def create_lat_bins_camchem_2d(spc, pres_indx,pres,s_year, e_year, output_strs, avg_opt):
    
    df_comb = pd.DataFrame()
    df_comb_abs = pd.DataFrame()
    
    lat_indx = (lat >= -20) & (lat <= 20)
    label = 'Tropical'
    
    df = create_df_camchem_2d(spc, pres_indx, lat_indx, label, s_year, e_year, output_strs, avg_opt)
    df_comb = pd.concat([df_comb, df])

    
    lat_indx = ((lat <-20) & (lat >-60)) 
    label = 'S Extratrop'
    df = create_df_camchem_2d(spc, pres_indx, lat_indx, label, s_year, e_year, output_strs, avg_opt)
    df_comb = pd.concat([df_comb, df])

    
    lat_indx = (lat > 20) & (lat < 60)
    label = 'N Extratrop'
    df = create_df_camchem_2d(spc, pres_indx, lat_indx, label, s_year, e_year, output_strs, avg_opt)
    df_comb = pd.concat([df_comb, df])

    
    filepath = '/home/qdzhu/fs09/Stratocu-planet/Chemistry/Output/dfs/'
    filename = '{}_{}_pres_{}_avg_{}.csv'.format(output_strs[0], spc,pres, avg_opt)
    df_comb.to_csv(filepath + filename)

def calculate_cam_chem_2d(spc, output_strs, s_year, e_year):
    nyear = e_year - s_year
    mass_trop = np.zeros((len(output_strs),nyear,192,288))
    volumes = np.zeros((len(output_strs),nyear,192,288))
    ohs = np.zeros((len(output_strs),nyear,192,288))
    for i_output, output_str in enumerate(output_strs):
        for i_year, year in enumerate(range(s_year,e_year)):
            oh = np.load('../Output/camchem_{}_3d/{}_year_{}.npy'.format(output_str,spc,year))[:,:]
                
            ohs[i_output,i_year,:,:] = oh
    filename = '../Output/camchem_{}_3d/{}_comb_eyear_{}.npy'.format(output_str, spc, e_year)
    np.save(filename, np.nanmean(ohs,axis=1))

    filename = '../Output/camchem_{}_3d/{}_comb_tot_eyear_{}.npy'.format(output_str, spc, e_year)
    np.save(filename, ohs)
    
def calculate_aqua_chem_2d(spc, output_strs, s_year, e_year):
    nyear = e_year - s_year
    aqua_mass_trop = np.zeros((len(output_strs),nyear,12,192,288))
    aqua_volumes = np.zeros((len(output_strs),nyear,12,192,288))
    aqua_spcs = np.zeros((len(output_strs),nyear,12,192,288))

    for i_output, output_str in enumerate(output_strs):
        for i_year, year in enumerate(range(s_year,e_year)):
            
            oh = np.load('../Output/aquas_{}_3d/{}_month_year_{}.npy'.format(output_str,spc,year))[:,:,:]
            aqua_spcs[i_output,i_year,:,:,:] = oh

    aqua_spcs = aqua_spcs.reshape((len(output_strs),nyear*12,192,288))

    #aqua_mass_trop = aqua_mass_trop[:,24:,:,:]
    #aqua_volumes = aqua_volumes[:,24:,:,:]
    #aqua_spcs = aqua_spcs[:,24:,:,:]
    aqua_mass_trop = aqua_mass_trop[:,12:,:,:]
    aqua_volumes = aqua_volumes[:,12:,:,:]
    aqua_spcs = aqua_spcs[:,12:,:,:]
    
    filename = '../Output/aquas_{}_3d/{}_comb.npy'.format(output_str, spc)
    np.save(filename, np.nanmean(aqua_spcs,axis=1))
    
    filename = '../Output/aquas_{}_3d/{}_comb_tot.npy'.format(output_str, spc)
    np.save(filename, aqua_spcs)
    
def calculate_cam_chem(spc, output_strs, s_year, e_year):
    nyear = e_year - s_year
    mass_trop = np.zeros((len(output_strs),nyear,24,192,288))
    volumes = np.zeros((len(output_strs),nyear,24,192,288))
    ohs = np.zeros((len(output_strs),nyear,24,192,288))
    for i_output, output_str in enumerate(output_strs):
        for i_year, year in enumerate(range(s_year,e_year)):
            mass = np.load('../Output/camchem_{}_3d/{}_year_{}.npy'.format(output_str,'MASS_trop',year))[8:,:,:]
            mass_trop[i_output,i_year,:,:,:] = mass

            volume = np.load('../Output/camchem_{}_3d/{}_year_{}.npy'.format(output_str,'volume_full',year))[8:,:,:]
            volumes[i_output,i_year,:,:,:] = volume

            density = np.load('../Output/camchem_{}_3d/{}_year_{}.npy'.format(output_str,'density',year))[8:,:,:]
            
            oh = np.load('../Output/camchem_{}_3d/{}_year_{}.npy'.format(output_str,spc,year))[8:,:,:]
            if spc == 'CH4':
                oh = oh*6.02e23/(28.9643*1e-3)*density*1e-6*volume/(6.02e23)*18*1e-12
            if spc == 'CH4_CHML':
                oh = oh*volume/(6.02e23)*18*1e-12*86400*365
                
            ohs[i_output,i_year,:,:,:] = oh


    filename = '../Output/camchem_{}_3d/{}_comb_eyear_{}.npy'.format(output_str, spc, e_year)
    np.save(filename, np.nanmean(ohs,axis=1))

    filename = '../Output/camchem_{}_3d/{}_comb_tot_eyear_{}.npy'.format(output_str, spc, e_year)
    np.save(filename, ohs)

    
def calculate_aqua_chem(spc, output_strs, s_year, e_year):
    nyear = e_year - s_year
    aqua_mass_trop = np.zeros((len(output_strs),nyear,12,24,192,288))
    aqua_volumes = np.zeros((len(output_strs),nyear,12,24,192,288))
    aqua_spcs = np.zeros((len(output_strs),nyear,12,24,192,288))

    for i_output, output_str in enumerate(output_strs):
        for i_year, year in enumerate(range(s_year,e_year)):
            mass = np.load('../Output/aquas_{}_3d/{}_month_year_{}.npy'.format(output_str,'MASS_trop',year))[:,8:,:,:]
            aqua_mass_trop[i_output,i_year,:,:,:,:] = mass
            volume = np.load('../Output/aquas_{}_3d/{}_month_year_{}.npy'.format(output_str,'volume_full',year))[:,8:,:,:]
            aqua_volumes[i_output,i_year,:,:,:,:] = volume

            density = np.load('../Output/aquas_{}_3d/{}_month_year_{}.npy'.format(output_str,'density',year))[:,8:,:,:]
            oh = np.load('../Output/aquas_{}_3d/{}_month_year_{}.npy'.format(output_str,spc,year))[:,8:,:,:]
            if spc == 'CH4':
                oh = oh*6.02e23/(28.9643*1e-3)*density*1e-6*volume/(6.02e23)*16*1e-12
            if spc == 'CH4_CHML': #molec cm-3 s-1 * volume (cm3) /(6.02e23) *16
                oh = oh*volume/(6.02e23)*16*1e-12*86400*365
            aqua_spcs[i_output,i_year,:,:,:,:] = oh

    aqua_mass_trop = aqua_mass_trop.reshape((len(output_strs),nyear*12,24,192,288))
    aqua_volumes = aqua_volumes.reshape((len(output_strs),nyear*12,24,192,288))
    aqua_spcs = aqua_spcs.reshape((len(output_strs),nyear*12,24,192,288))

    #aqua_mass_trop = aqua_mass_trop[:,24:,:,:,:]
    #aqua_volumes = aqua_volumes[:,24:,:,:,:]
    #aqua_spcs = aqua_spcs[:,24:,:,:,:]
    aqua_mass_trop = aqua_mass_trop[:,12:,:,:,:]
    aqua_volumes = aqua_volumes[:,12:,:,:,:]
    aqua_spcs = aqua_spcs[:,12:,:,:,:]
    
    filename = '../Output/aquas_{}_3d/{}_comb.npy'.format(output_str, spc)
    np.save(filename, np.nanmean(aqua_spcs,axis=1))
    
    filename = '../Output/aquas_{}_3d/{}_comb_tot.npy'.format(output_str, spc)
    np.save(filename, aqua_spcs)

def calculate_vert_map(output_str, spc):
    tropopause = np.load('/home/qdzhu/fs09/Stratocu-planet/Chemistry/Output/vert_map/tropopause.npy')
    if 'FC2000' in output_str:
        aqua_ch4 = np.load('../Output/camchem_{}_3d/{}_comb_tot.npy'.format(output_str,spc))
        aqua_volume = np.load('../Output/camchem_{}_3d/{}_comb_tot.npy'.format(output_str,'volume_full'))
    else:
        aqua_ch4 = np.load('../Output/aquas_{}_3d/{}_comb_tot.npy'.format(output_str,spc))
        aqua_volume = np.load('../Output/aquas_{}_3d/{}_comb_tot.npy'.format(output_str,'volume_full'))
    #set nan values above tropopause
    for i in range(192):
        indx = lev<tropopause[i]
        aqua_ch4[:,:,indx,i,:] = np.nan
        aqua_volume[:,:,indx,i,:] = np.nan
    aqua_oh_vert = np.nansum(aqua_ch4*aqua_volume,axis=4)[0,:,:,:]/np.nansum(aqua_volume,axis=4)[0,:,:,:]
    aqua_oh_trop = np.nansum(aqua_ch4*aqua_volume,axis=2)[0,:,:,:]/np.nansum(aqua_volume,axis=2)[0,:,:,:]
    ut_indx = lev<400
    aqua_oh_ut_trop = np.nansum(aqua_ch4[:,:,ut_indx,:,:]*aqua_volume[:,:,ut_indx,:,:],axis=2)[0,:,:,:]/np.nansum(aqua_volume[:,:,ut_indx,:,:],axis=2)[0,:,:,:]
    
    filepath = '/home/qdzhu/fs09/Stratocu-planet/Chemistry/Output/vert_map/'
    np.save(filepath + '{}_{}_vert.npy'.format(output_str, spc), np.nanmean(aqua_oh_vert,axis=0))
    np.save(filepath + '{}_{}_trop.npy'.format(output_str, spc), np.nanmean(aqua_oh_trop,axis=0))
    np.save(filepath + '{}_{}_ut_trop.npy'.format(output_str, spc), np.nanmean(aqua_oh_ut_trop,axis=0))
    
    np.save(filepath + '{}_{}_vert_std.npy'.format(output_str, spc), np.nanstd(aqua_oh_vert,axis=0))
    np.save(filepath + '{}_{}_trop_std.npy'.format(output_str, spc), np.nanstd(aqua_oh_trop,axis=0))
    np.save(filepath + '{}_{}_ut_trop_std.npy'.format(output_str, spc), np.nanstd(aqua_oh_ut_trop,axis=0))


def main(args):
    pres = args.p
    opt = args.o
    output1 = args.f
    output2 = args.s
    model_label = args.m
    avg_opt = args.v
    
    if pres == 'l':
        pres_indx = lev>=600
    elif pres == 't':
        pres_indx = lev>=200
    elif pres == 'tt':
        pres_indx = lev>=350
    print(output1)
    if opt == 'spc_2d':
        spcs_2d = ['O3_tot_column','LNO_COL_PROD']
        for spc in spcs_2d:
            output_strs = [output1]
            if output1 == 'FC2000':
                print(output1)
                s_year = 2
                e_year = 4
                create_lat_bins_camchem_2d(spc, pres_indx, pres,s_year, e_year, output_strs, avg_opt)
            else:
                print(output1)
                s_year = 1
                e_year = 6#change for LNO 6
                create_lat_bins_aquachem_2d(spc, pres_indx, pres,output_strs[0], s_year, e_year, output_strs, avg_opt)
    
    if opt == 'comb_2d':
        if 'FC2000' in output1:
            s_year = 2
            e_year = 4
            spcs = ['O3_tot_column','LNO_COL_PROD','CLDTOT','CLDTOT','CLDHGH','CLDMED','CLDLOW']
            #spcs = []
            for spc in spcs:
                calculate_cam_chem_2d(spc, [output1], s_year, e_year)
        else:
            s_year = 1
            e_year = 6
            spcs = ['O3_tot_column','LNO_COL_PROD','CLDTOT']
            for spc in spcs:
                calculate_aqua_chem_2d(spc, [output1], s_year, e_year)
                
    if opt == 'comb':
        if 'FC2000' in output1:
            s_year = 2
            e_year = 32
            spcs = ['OH','CH4','CH4_CHML','H2O','CO','O3','ISOP','jo3_a','NO','NO2','PAN','MASS_trop','volume','LNO_PROD','volume_full','OMEGA','T']
            spcs = ['O3_Loss','O3_Prod']
            for spc in spcs:
                calculate_cam_chem(spc, [output1], s_year, e_year)
        else:
            s_year = 1
            e_year = 6
            spcs = ['OH','CH4','CH4_CHML','H2O','CO','O3','ISOP','jo3_a','NO','NO2','PAN','MASS_trop','volume','LNO_PROD','volume_full']
            #spcs = ['T']
            spcs = ['OMEGA']
            for spc in spcs:
                calculate_aqua_chem(spc, [output1], s_year, e_year)
                
    if opt == 'vert_map':
        #spcs = ['OH','CH4_CHML','H2O','CO','O3','NO','NO2','volume_full']
        spcs =['OH','CH4','CH4_CHML','H2O','CO','O3','ISOP','jo3_a','NO','NO2','PAN','MASS_trop','volume','LNO_PROD','volume_full']
        for spc in spcs:
            calculate_vert_map(output1, spc)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read in index of CESM2 model')
    parser.add_argument('-p', type=str, help='Specify the pressure')
    parser.add_argument('-o', type=str, help='Specify the option')
    
    parser.add_argument('-f', type=str, help='Specify the reference')
    parser.add_argument('-s', type=str, help='Specify the increment')
    parser.add_argument('-m', type=str, help='Specify the model_label')
    parser.add_argument('-v', type=str, help='Specify the avg_opt')
    args = parser.parse_args()
    main(args)