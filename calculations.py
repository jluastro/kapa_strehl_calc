import pandas
import numpy as np
import pylab as plt
import base64
from astropy import units as u
from astropy import constants as c
from astropy import table
from scipy import interpolate as interp
import time
import pdb

all_ref_filters = ['V', 'r', 'i', 'J', 'H', 'K']
all_spec_types = ['K']

def load_data_tables():
    # Setup the tables for our Strehl Calculation
    # Read in the table of Strehl and FWHM results
    tab = table.Table.read('strehl_grid.fits')

    # Clean out any bad data.
    idx = np.where(tab['fwhm'][:, -1] != 0)[0]

    tab = tab[idx]

    # For now, lets also drop the tt_offset >= 21 since that grid point wasn't complete.
    idx = np.where(tab['tt_offset'] < 21)[0]
    tab = tab[idx]

    # Sort the table so we ensure that the columns will iterate in the right order.
    tab.sort(['zenith_angle', 'sci_exp_time', 'tt_offset', 'tt_vmag'])

    return tab


def prep_grid(tab):
    uni_zenith = np.unique(tab['zenith_angle'])
    uni_s_texp = np.unique(tab['sci_exp_time'])
    uni_ttvmag = np.unique(tab['tt_vmag'])
    uni_ttoffs = np.unique(tab['tt_offset'])

    # Prep a table for each reference filter, TT spectral type combo.
    points_dict = {}
    gr_strehl_dict = {}
    gr_fwhm_dict = {}
    gr_ttm_dict = {}
    
    for ff in range(len(all_ref_filters)):
        for ss in range(len(all_spec_types)):
            filt_name = all_ref_filters[ff]
            type_name = all_spec_types[ss]

            dict_idx = filt_name + ',' + type_name

            idx = np.where(tab['tt_spec_type'] == type_name)[0]

            tab_tmp = tab[idx]
        
            if filt_name == 'V':
                uni_ttmag = np.unique(tab_tmp['tt_vmag'])
            else:
                # Note I have hard-coded to match the order of the filters
                # at the time of generation.
                ff_in_tt_mag = ff - 1
                uni_ttmag = np.unique(tab_tmp['tt_mag'][:, ff_in_tt_mag])
            
            points = (uni_zenith, uni_s_texp, uni_ttoffs, uni_ttmag)

            N_za = len(uni_zenith)
            N_texp = len(uni_s_texp)
            N_ttm = len(uni_ttmag)
            N_tto = len(uni_ttoffs)
            N_filt = len(tab['strehl'][0, :])

            # gr_za = tab['zenith_angle'].reshape(N_za, N_texp, N_tto, N_ttv)
            # gr_texp = tab['sci_exp_time'].reshape(N_za, N_texp, N_tto, N_ttv)
            # gr_ttv = tab['tt_vmag'].reshape(N_za, N_texp, N_tto, N_ttv)
            # gr_tto = tab['tt_offset'].reshape(N_za, N_texp, N_tto, N_ttv)
            gr_strehl = tab_tmp['strehl'].reshape(N_za, N_texp, N_tto, N_ttm, N_filt)
            gr_fwhm = tab_tmp['fwhm'].reshape(N_za, N_texp, N_tto, N_ttm, N_filt)
            gr_ttm = tab_tmp['tt_mag'].reshape(N_za, N_texp, N_tto, N_ttm, N_filt)

            points_dict[dict_idx] = points
            gr_strehl_dict[dict_idx] = gr_strehl
            gr_fwhm_dict[dict_idx] = gr_fwhm
            gr_ttm_dict[dict_idx] = gr_ttm


    return points_dict, gr_strehl_dict, gr_fwhm_dict, gr_ttm_dict


def calc_performance(user_inputs, grid_inputs):
    in_za = user_inputs[0]
    in_texp = user_inputs[1]
    in_tto = user_inputs[2]
    in_tt_ref_mag = user_inputs[3]
    in_tt_ref_filt = user_inputs[4]
    in_tt_type = user_inputs[5]
    in_tt_band = user_inputs[6]

    in_points = np.array([float(in_za), float(in_texp), float(in_tto), float(in_tt_ref_mag)])
    in_dict_idx = in_tt_ref_filt + ',' + in_tt_type

    points = grid_inputs[0][in_dict_idx]
    gr_strehl = grid_inputs[1][in_dict_idx]
    gr_fwhm = grid_inputs[2][in_dict_idx]
    gr_ttm = grid_inputs[3][in_dict_idx]

    N_filt = gr_strehl.shape[4]
    
    out_strehl = np.zeros(N_filt, dtype=float)
    out_fwhm = np.zeros(N_filt, dtype=float)
    out_ttm = np.zeros(N_filt, dtype=float)

    for ff in range(N_filt):
        int_strehl = interp.RegularGridInterpolator(points, gr_strehl[:, :, :, :, ff], method='linear', bounds_error=False)
        int_fwhm = interp.RegularGridInterpolator(points, gr_fwhm[:, :, :, :, ff], method='linear', bounds_error=False)
        int_ttm = interp.RegularGridInterpolator(points, gr_ttm[:, :, :, :, ff], method='linear', bounds_error=False)
        
        out_strehl[ff] = int_strehl(in_points)
        out_fwhm[ff] = int_fwhm(in_points)
        out_ttm[ff] = int_ttm(in_points)

    return out_strehl, out_fwhm, out_ttm

