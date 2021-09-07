import pandas
import numpy as np
import pylab as plt
import base64
from astropy import units as u
from astropy import constants as c
from astropy import table
from scipy import interpolate as interp
import time
import calculations as calc


####################
# Test #1
####################
def test1():
    tab = calc.load_data_tables()

    grid_inputs = calc.prep_grid(tab)

    in_za = 30.0
    in_texp = 30.0
    in_tto = 0.0
    in_tt_ref_mag = 10.0
    in_tt_ref_filt = 'V'
    in_tt_type = 'K'
    in_tt_band = 'R'

    user_inputs = np.array([in_za, in_texp, in_tto, in_tt_ref_mag, in_tt_ref_filt, in_tt_type, in_tt_band])
    
    strehl, fwhm, ttm = calc.calc_performance(user_inputs, grid_inputs)

    strehl_good = np.array([0.09, 0.16, 0.25, 0.41, 0.57])
    fwhm_good = np.array([70.9, 61.8, 38.0, 42.2, 49.4])
    ttm_good = np.array([8.76, 6.66, 7.50, 6.80, 6.60])

    np.testing.assert_almost_equal(strehl, strehl_good, decimal=2)
    np.testing.assert_almost_equal(fwhm, fwhm_good, decimal=2)
    np.testing.assert_almost_equal(ttm, ttm_good, decimal=2)
    
    return


def test2(tt_ref_filt):
    tab = calc.load_data_tables()

    grid_inputs = calc.prep_grid(tab)

    in_za = 30.0
    in_texp = 30.0
    in_tto = 0.0
    in_tt_ref_mag = 15.0
    in_tt_ref_filt = tt_ref_filt
    in_tt_type = 'K'
    in_tt_band = 'R'

    user_inputs = np.array([in_za, in_texp, in_tto, in_tt_ref_mag, in_tt_ref_filt, in_tt_type, in_tt_band])
    
    strehl, fwhm, ttm = calc.calc_performance(user_inputs, grid_inputs)
    print(strehl, fwhm, ttm)

    return

