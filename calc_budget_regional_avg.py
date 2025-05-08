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

budget_term =  ['OH_CHMP','OH_CHML'] 

#CH4
ch4_lterm = ['r_CH4_OH']
#CO
co_lterm = ['r_CO_OH_M','r_CO_OH_b']
#H2, O3, H2O2, radical–radical reactions
hoy_lterm = ['r_OH_H2', 'r_OH_H2O2', 'r_OH_HO2','r_OH_O3', 'r_OH_O']
# NOy
noy_lterm = ['r_NO2_OH','r_NH3_OH', 'r_NOA_OH']
#c1-VOC
c1_lterm = ['r_CH2O_OH', 'r_CH3OH_OH', 'r_CH3OOH_OH', 'r_HCOOH_OH']
# bio
bio_lterm = ['r_ISOP_OH','r_MTERP_OH']
#c2-VOC
c2_lterm = ['r_C2H2_OH_M', 'r_C2H5OH_OH', 'r_C2H5OOH_OH', 'r_C2H6_OH',
            'r_CH3CHO_OH', 'r_CH3COOH_OH', 'r_CH3COOOH_OH', 'r_CH3CN_OH', 
            'r_GLYALD_OH', 'r_GLYOXAL_OH', 'r_PAN_OH', 'r_C3H7OOH_OH',
            'r_C3H8_OH', 'r_CH3COCHO_OH', 'r_ROOH_OH', 'r_BIGENE_OH',
            'r_MEK_OH', 'r_MEKOOH_OH', 'r_MACROOH_OH', 'r_ALKOOH_OH',
            'r_BIGALK_OH', 'r_HPALD_OH', 'r_HYDRALD_OH', 'r_IEPOX_OH',
            'r_ISOPOOH_OH', 'r_TOLOOH_OH', 'r_XYLENOOH_OH', 'r_NTERPOOH_OH',
            'r_TERP2OOH_OH', 'r_C2H4_OH']

pterms = ['r_HO2_O3','r_O1D_H2O', 'r_NO_HO2','jh2o2']

budget_labels = ['OH PROD','OH Loss','CH$_4$+OH','CO+OH','HOy+OH',
          'ISOP+OH','C1VOC+OH','C2VOC+OH','NOy+OH',
          'O$_3$+HO$_2$','2x(O$^1$D+H$_2$O)','NO+HO$_2$','2xj(H$_2$O$_2$)']
cls_labels = ['Prod Total','Loss Total','Loss','Loss','Loss',
             'Loss','Loss','Loss','Loss',
             'Prod','Prod','Prod','Prod']
cls_labels = np.array(cls_labels)

data = xr.open_dataset('/net/fs09/d0/qdzhu/result/Nicole_aquaQPC6_f09_AveChem_LBC_IC/run/AveChem_LBC_IC.i.nc')
lat = data['lat'].values
lon = data['lon'].values
lev = data['lev'].values
lev = lev[8:]

def calculate_oh_prod_loss_online(output_str, year):
    if 'FC2000' in output_str:
        filestr = 'camchem'
    else:
        filestr = 'aquas'
    spc_comb = budget_term + ch4_lterm
    budget_comb = np.zeros((len(spc_comb),24,192,288))
    for i_spc in range(len(spc_comb)):
        this_spc = np.load('../Output/{}_{}_3d/{}_year_{}.npy'.format(filestr,output_str,spc_comb[i_spc],year))
        this_spc = this_spc[8:,:,:]
        budget_comb[i_spc,:,:,:] = this_spc
    
    spc_comb = co_lterm
    comb = np.zeros((len(spc_comb),24,192,288))
    for i_spc in range(len(spc_comb)):
        this_spc = np.load('../Output/{}_{}_3d/{}_year_{}.npy'.format(filestr,output_str,spc_comb[i_spc],year))
        this_spc = this_spc[8:,:,:]
        comb[i_spc,:,:,:] = this_spc
    lco = np.sum(comb, axis=0)
    
    spc_comb = hoy_lterm
    comb = np.zeros((len(spc_comb),24,192,288))
    for i_spc in range(len(spc_comb)):
        this_spc = np.load('../Output/{}_{}_3d/{}_year_{}.npy'.format(filestr,output_str,spc_comb[i_spc],year))
        this_spc = this_spc[8:,:,:]
        comb[i_spc,:,:,:] = this_spc
    lhoy = np.sum(comb, axis=0)
    
    spc_comb = noy_lterm
    comb = np.zeros((len(spc_comb),24,192,288))
    for i_spc in range(len(spc_comb)):
        this_spc = np.load('../Output/{}_{}_3d/{}_year_{}.npy'.format(filestr,output_str,spc_comb[i_spc],year))
        this_spc = this_spc[8:,:,:]
        comb[i_spc,:,:,:] = this_spc
    lnoy = np.sum(comb, axis=0)
    
    spc_comb = c1_lterm
    comb = np.zeros((len(spc_comb),24,192,288))
    for i_spc in range(len(spc_comb)):
        this_spc = np.load('../Output/{}_{}_3d/{}_year_{}.npy'.format(filestr,output_str,spc_comb[i_spc],year))
        this_spc = this_spc[8:,:,:]
        comb[i_spc,:,:,:] = this_spc
    lc1 = np.sum(comb, axis=0)
    
    spc_comb = c2_lterm
    comb = np.zeros((len(spc_comb),24,192,288))
    for i_spc in range(len(spc_comb)):
        this_spc = np.load('../Output/{}_{}_3d/{}_year_{}.npy'.format(filestr,output_str,spc_comb[i_spc],year))
        this_spc = this_spc[8:,:,:]
        comb[i_spc,:,:,:] = this_spc
    lc2 = np.sum(comb, axis=0)
    
    spc_comb = bio_lterm
    comb = np.zeros((len(spc_comb),24,192,288))
    for i_spc in range(len(spc_comb)):
        this_spc = np.load('../Output/{}_{}_3d/{}_year_{}.npy'.format(filestr,output_str,spc_comb[i_spc],year))
        this_spc = this_spc[8:,:,:]
        comb[i_spc,:,:,:] = this_spc
    lbio = np.sum(comb, axis=0)
    
    ptot = budget_comb[0,:,:,:]
    ltot = budget_comb[1,:,:,:]
    lch4 = budget_comb[2,:,:,:]
    
   
    spc_comb = pterms
    pcomb = np.zeros((len(spc_comb),24,192,288))
    for i_spc in range(len(spc_comb)):
        #print(spc_comb[i_spc])
        this_spc = np.load('../Output/{}_{}_3d/{}_year_{}.npy'.format(filestr,output_str,spc_comb[i_spc],year))
        this_spc = this_spc[8:,:,:]
        
        pcomb[i_spc,:,:,:] = this_spc
        if spc_comb[i_spc] == 'r_O1D_H2O':
            pcomb[i_spc,:,:,:] = pcomb[i_spc,:,:,:]*2
        if spc_comb[i_spc] == 'jh2o2':
            h2o2 = np.load('../Output/{}_{}_3d/{}_year_{}.npy'.format(filestr,output_str,'H2O2',year))
            h2o2 = h2o2[8:,:,:]
            density = np.load('../Output/{}_{}_3d/{}_year_{}.npy'.format(filestr,output_str,'density',year))
            density = density[8:,:,:]
            h2o2 = h2o2*6.02e23/(28.9643*1e-3)*density*1e-6
            pcomb[i_spc,:,:,:] = 2*this_spc*h2o2
        
    budget_tot_comb = np.zeros((13,24,192,288))
    budget_tot_comb[:3,:,:,:] = budget_comb
    budget_tot_comb[3,:,:,:] = lco
    budget_tot_comb[4,:,:,:] = lhoy
    budget_tot_comb[5,:,:,:] = lbio
    budget_tot_comb[6,:,:,:] = lc1
    budget_tot_comb[7,:,:,:] = lc2
    budget_tot_comb[8,:,:,:] = lnoy
    budget_tot_comb[9:13,:,:,:] = pcomb
    
    return budget_tot_comb

def calculate_aqua_oh_prod_loss_online(output_str, year):
    if 'FC2000' in output_str:
        filestr = 'camchem'
    else:
        filestr = 'aquas'
    spc_comb = budget_term + ch4_lterm
    budget_comb = np.zeros((len(spc_comb),12,24,192,288))
    for i_spc in range(len(spc_comb)):
        this_spc = np.load('../Output/{}_{}_3d/{}_month_year_{}.npy'.format(filestr,output_str,spc_comb[i_spc],year))
        this_spc = this_spc[:,8:,:,:]
        budget_comb[i_spc,:,:,:,:] = this_spc
    
    spc_comb = co_lterm
    comb = np.zeros((len(spc_comb),12,24,192,288))
    for i_spc in range(len(spc_comb)):
        this_spc = np.load('../Output/{}_{}_3d/{}_month_year_{}.npy'.format(filestr,output_str,spc_comb[i_spc],year))
        this_spc = this_spc[:,8:,:,:]
        comb[i_spc,:,:,:,:] = this_spc
    lco = np.sum(comb, axis=0)
    
    spc_comb = hoy_lterm
    comb = np.zeros((len(spc_comb),12,24,192,288))
    for i_spc in range(len(spc_comb)):
        this_spc = np.load('../Output/{}_{}_3d/{}_month_year_{}.npy'.format(filestr,output_str,spc_comb[i_spc],year))
        this_spc = this_spc[:,8:,:,:]
        comb[i_spc,:,:,:,:] = this_spc
    lhoy = np.sum(comb, axis=0)
    
    spc_comb = noy_lterm
    comb = np.zeros((len(spc_comb),12,24,192,288))
    for i_spc in range(len(spc_comb)):
        this_spc = np.load('../Output/{}_{}_3d/{}_month_year_{}.npy'.format(filestr,output_str,spc_comb[i_spc],year))
        this_spc = this_spc[:,8:,:,:]
        comb[i_spc,:,:,:,:] = this_spc
    lnoy = np.sum(comb, axis=0)
    
    spc_comb = c1_lterm
    comb = np.zeros((len(spc_comb),12,24,192,288))
    for i_spc in range(len(spc_comb)):
        this_spc = np.load('../Output/{}_{}_3d/{}_month_year_{}.npy'.format(filestr,output_str,spc_comb[i_spc],year))
        this_spc = this_spc[:,8:,:,:]
        comb[i_spc,:,:,:,:] = this_spc
    lc1 = np.sum(comb, axis=0)
    
    spc_comb = c2_lterm
    comb = np.zeros((len(spc_comb),12,24,192,288))
    for i_spc in range(len(spc_comb)):
        this_spc = np.load('../Output/{}_{}_3d/{}_month_year_{}.npy'.format(filestr,output_str,spc_comb[i_spc],year))
        this_spc = this_spc[:,8:,:,:]
        comb[i_spc,:,:,:,:] = this_spc
    lc2 = np.sum(comb, axis=0)
    
    spc_comb = bio_lterm
    comb = np.zeros((len(spc_comb),12,24,192,288))
    for i_spc in range(len(spc_comb)):
        this_spc = np.load('../Output/{}_{}_3d/{}_month_year_{}.npy'.format(filestr,output_str,spc_comb[i_spc],year))
        this_spc = this_spc[:,8:,:,:]
        comb[i_spc,:,:,:,:] = this_spc
    lbio = np.sum(comb, axis=0)
    
    ptot = budget_comb[0,:,:,:,:]
    ltot = budget_comb[1,:,:,:,:]
    lch4 = budget_comb[2,:,:,:,:]
    
   
    spc_comb = pterms
    pcomb = np.zeros((len(spc_comb),12,24,192,288))
    for i_spc in range(len(spc_comb)):
        #print(spc_comb[i_spc])
        this_spc = np.load('../Output/{}_{}_3d/{}_month_year_{}.npy'.format(filestr,output_str,spc_comb[i_spc],year))
        this_spc = this_spc[:,8:,:,:]
        
        pcomb[i_spc,:,:,:,:] = this_spc
        if spc_comb[i_spc] == 'r_O1D_H2O':
            pcomb[i_spc,:,:,:,:] = pcomb[i_spc,:,:,:,:]*2
        if spc_comb[i_spc] == 'jh2o2':
            h2o2 = np.load('../Output/{}_{}_3d/{}_month_year_{}.npy'.format(filestr,output_str,'H2O2',year))
            h2o2 = h2o2[:,8:,:,:]
            density = np.load('../Output/{}_{}_3d/{}_month_year_{}.npy'.format(filestr,output_str,'density',year))
            density = density[:,8:,:,:]
            h2o2 = h2o2*6.02e23/(28.9643*1e-3)*density*1e-6
            pcomb[i_spc,:,:,:,:] = 2*this_spc*h2o2
        
    budget_tot_comb = np.zeros((13,12,24,192,288))
    budget_tot_comb[:3,:,:,:] = budget_comb
    budget_tot_comb[3,:,:,:] = lco
    budget_tot_comb[4,:,:,:] = lhoy
    budget_tot_comb[5,:,:,:] = lbio
    budget_tot_comb[6,:,:,:] = lc1
    budget_tot_comb[7,:,:,:] = lc2
    budget_tot_comb[8,:,:,:] = lnoy
    budget_tot_comb[9:13,:,:,:] = pcomb
    
    return budget_tot_comb


def calculate_cam_chem(output_strs, s_year, e_year):
    spc = 'OH'
    nyear = e_year - s_year
    ohs = np.zeros((len(output_strs),nyear,24,192,288))
    budget = np.zeros((len(output_strs),13,nyear, 24,192,288))
    for i_output, output_str in enumerate(output_strs):
        for i_year, year in enumerate(range(s_year,e_year)):
            oh = np.load('../Output/camchem_{}_3d/{}_year_{}.npy'.format(output_str,spc,year))[8:,:,:]
            ohs[i_output,i_year,:,:,:] = oh
            budget[i_output,:,i_year,:,:,:] = calculate_oh_prod_loss_online(output_str, year)
    
    filename = '../Output/camchem_{}_3d/OH_budget_comb_tot.npy'.format(output_str)
    np.save(filename, budget)
    
    filename = '../Output/camchem_{}_3d/OH_budget_comb.npy'.format(output_str)
    np.save(filename, np.nanmean(budget,axis=2))
        
    budget_freq = np.zeros(budget.shape)
    for i in range(budget.shape[0]):
        budget_freq[i,:,:,:,:,:] =budget[i,:,:,:,:,:]/ohs

    filename = '../Output/camchem_{}_3d/OH_budget_freq_comb_tot.npy'.format(output_str)
    np.save(filename, budget_freq)
    
    filename = '../Output/camchem_{}_3d/OH_budget_freq_comb.npy'.format(output_str)
    np.save(filename, np.nanmean(budget_freq,axis=2))

def calculate_aqua_chem(output_strs, s_year, e_year):
    nyear = e_year - s_year
    aqua_mass_trop = np.zeros((len(output_strs),nyear,12,24,192,288))
    aqua_volumes = np.zeros((len(output_strs),nyear,12,24,192,288))
    aqua_spcs = np.zeros((len(output_strs),nyear,12,24,192,288))
    aqua_budget = np.zeros((len(output_strs),13,nyear,12,24,192,288))
    spc = 'OH'
    for i_output, output_str in enumerate(output_strs):
        for i_year, year in enumerate(range(s_year,e_year)):
            density = np.load('../Output/aquas_{}_3d/{}_month_year_{}.npy'.format(output_str,'density',year))[:,8:,:,:]
            oh = np.load('../Output/aquas_{}_3d/{}_month_year_{}.npy'.format(output_str,spc,year))[:,8:,:,:]
            aqua_spcs[i_output,i_year,:,:,:,:] = oh
            
            aqua_budget[i_output,:,i_year,:,:,:,:] = calculate_aqua_oh_prod_loss_online(output_str, year)

    aqua_spcs = aqua_spcs.reshape((len(output_strs),nyear*12,24,192,288))
    aqua_budget = aqua_budget.reshape((len(output_strs),13,nyear*12,24,192,288))
    #aqua_spcs = aqua_spcs[:,24:,:,:,:]
    #aqua_budget = aqua_budget[:,:,24:,:,:,:]
    
    aqua_spcs = aqua_spcs[:,12:,:,:,:]
    aqua_budget = aqua_budget[:,:,12:,:,:,:]
    
    filename = '../Output/aquas_{}_3d/OH_budget_comb_tot.npy'.format(output_str)
    np.save(filename, aqua_budget)
    
    filename = '../Output/aquas_{}_3d/OH_budget_comb.npy'.format(output_str)
    np.save(filename, np.nanmean(aqua_budget,axis=2))

    aqua_budget_freq = np.zeros(aqua_budget.shape)
    for i in range(aqua_budget.shape[0]):
        aqua_budget_freq[i,:,:,:,:,:] = aqua_budget[i,:,:,:,:,:]/aqua_spcs

    filename = '../Output/aquas_{}_3d/OH_budget_freq_comb_tot.npy'.format(output_str)
    np.save(filename, aqua_budget_freq)
    
    filename = '../Output/aquas_{}_3d/OH_budget_freq_comb.npy'.format(output_str)
    np.save(filename, np.nanmean(aqua_budget_freq,axis=2))

    
def calculate_budget_vert_map(output_str):
    tropopause = np.load('/home/qdzhu/fs09/Stratocu-planet/Chemistry/Output/vert_map/tropopause.npy')
    if 'FC2000' in output_str:
        aqua_budget = np.load('../Output/camchem_{}_3d/OH_budget_comb.npy'.format(output_str))
        aqua_oh = np.load('../Output/camchem_{}_3d/OH_comb.npy'.format(output_str))
        aqua_volume = np.load('../Output/camchem_{}_3d/{}_comb.npy'.format(output_str,'volume_full'))
    else:
        aqua_budget = np.load('../Output/aquas_{}_3d/OH_budget_comb.npy'.format(output_str))
        aqua_oh = np.load('../Output/aquas_{}_3d/OH_comb.npy'.format(output_str))
        aqua_volume = np.load('../Output/aquas_{}_3d/{}_comb.npy'.format(output_str,'volume_full'))
    #set nan values above tropopause
    for i in range(192):
        indx = lev<tropopause[i]
        aqua_oh[:,indx,i,:] = np.nan
        aqua_volume[:,indx,i,:] = np.nan
    
    aqua_budget_vert = np.zeros((13,24,192))
    aqua_budget_freq_vert = np.zeros((13,24,192))
    for i in range(aqua_budget_vert.shape[0]):
        aqua_budget_vert[i,:,:]= np.nansum(aqua_budget[0,i,:,:,:]*aqua_volume[0,:,:,:],axis=2)*1e-12*365*86400/(6.02e23)
        aqua_budget_freq_vert[i,:,:] = np.nansum(aqua_budget[0,i,:,:,:]/aqua_oh[0,:,:,:]*aqua_volume[0,:,:,:],axis=2)/np.nansum(aqua_volume[0,:,:,:],axis=2)

    
    aqua_budget_map = np.zeros((13,192,288))
    aqua_budget_freq_map = np.zeros((13,192,288))
    for i in range(aqua_budget_vert.shape[0]):
        aqua_budget_map[i,:,:]= np.nansum(aqua_budget[0,i,:,:,:]*aqua_volume[0,:,:,:],axis=0)*1e-12*365*86400/(6.02e23)
        aqua_budget_freq_map[i,:,:] = np.nansum(aqua_budget[0,i,:,:,:]/aqua_oh[0,:,:,:]*aqua_volume[0,:,:,:],axis=0)/np.nansum(aqua_volume[0,:,:,:],axis=0)


    ut_indx = lev<400
    
    aqua_budget_ut_map = np.zeros((13,192,288))
    aqua_budget_freq_ut_map = np.zeros((13,192,288))
    for i in range(aqua_budget_vert.shape[0]):
        aqua_budget_ut_map[i,:,:]= np.nansum(aqua_budget[0,i,ut_indx,:,:]*aqua_volume[0,ut_indx,:,:],axis=0)*1e-12*365*86400/(6.02e23)
        aqua_budget_freq_ut_map[i,:,:] = np.nansum(aqua_budget[0,i,ut_indx,:,:]/aqua_oh[0,ut_indx,:,:]*aqua_volume[0,ut_indx,:,:],axis=0)/np.nansum(aqua_volume[0,ut_indx,:,:],axis=0)


    filepath = '/home/qdzhu/fs09/Stratocu-planet/Chemistry/Output/vert_map/'
    np.save(filepath + '{}_{}_vert.npy'.format(output_str, 'budget'), aqua_budget_vert)
    np.save(filepath + '{}_{}_vert.npy'.format(output_str, 'budget_freq'), aqua_budget_freq_vert)
    
    np.save(filepath + '{}_{}_trop.npy'.format(output_str, 'budget'), aqua_budget_map)
    np.save(filepath + '{}_{}_trop.npy'.format(output_str, 'budget_freq'), aqua_budget_freq_map)
    
    np.save(filepath + '{}_{}_ut_trop.npy'.format(output_str, 'budget'), aqua_budget_ut_map)
    np.save(filepath + '{}_{}_ut_trop.npy'.format(output_str, 'budget_freq'), aqua_budget_freq_ut_map)


def main(args):
    opt = args.o
    output = args.f 

    if opt == 'budget':
        if 'FC2000' in output:
            s_year = 2
            e_year = 4#32
            calculate_cam_chem([output], s_year, e_year)
        else:
            s_year = 1
            e_year = 6
            calculate_aqua_chem([output], s_year, e_year)
            
    if opt == 'vert_map':
        calculate_budget_vert_map(output)
        
    
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