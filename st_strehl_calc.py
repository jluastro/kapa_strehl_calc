import streamlit as st
import pandas
import numpy as np
import pylab as plt
import base64
from astropy import units as u
from astropy import constants as c
from astropy import table
from scipy import interpolate as interp
import time


def load_data_table():
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
    points = (uni_zenith, uni_s_texp, uni_ttoffs, uni_ttvmag)

    N_za = len(uni_zenith)
    N_texp = len(uni_s_texp)
    N_ttv = len(uni_ttvmag)
    N_tto = len(uni_ttoffs)
    N_filt = len(tab['strehl'][0, :])

    # gr_za = tab['zenith_angle'].reshape(N_za, N_texp, N_tto, N_ttv)
    # gr_texp = tab['sci_exp_time'].reshape(N_za, N_texp, N_tto, N_ttv)
    # gr_ttv = tab['tt_vmag'].reshape(N_za, N_texp, N_tto, N_ttv)
    # gr_tto = tab['tt_offset'].reshape(N_za, N_texp, N_tto, N_ttv)
    gr_strehl = tab['strehl'].reshape(N_za, N_texp, N_tto, N_ttv, N_filt)
    gr_fwhm = tab['fwhm'].reshape(N_za, N_texp, N_tto, N_ttv, N_filt)
    gr_ttm = tab['tt_mag'].reshape(N_za, N_texp, N_tto, N_ttv, N_filt)

    return points, gr_strehl, gr_fwhm, gr_ttm


def calc_performance(user_inputs, grid_inputs):
    points = grid_inputs[0]
    gr_strehl = grid_inputs[1]
    gr_fwhm = grid_inputs[2]
    gr_ttm = grid_inputs[3]

    N_filt = gr_strehl.shape[4]
    
    out_strehl = np.zeros(N_filt, dtype=float)
    out_fwhm = np.zeros(N_filt, dtype=float)
    out_ttm = np.zeros(N_filt, dtype=float)

    for ff in range(N_filt):
        int_strehl = interp.RegularGridInterpolator(points, gr_strehl[:, :, :, :, ff], method='linear', bounds_error=False)
        int_fwhm = interp.RegularGridInterpolator(points, gr_fwhm[:, :, :, :, ff], method='linear', bounds_error=False)
        int_ttm = interp.RegularGridInterpolator(points, gr_ttm[:, :, :, :, ff], method='linear', bounds_error=False)
        
    
        out_strehl[ff] = int_strehl(user_inputs)
        out_fwhm[ff] = int_fwhm(user_inputs)
        out_ttm[ff] = int_ttm(user_inputs)

    return out_strehl, out_fwhm, out_ttm

##############################
#
# START Web App
#
##############################

##########
# Main Page
##########    
page_title = 'KAPA Strehl Calculator'
st.set_page_config(page_title = page_title, page_icon = ":eyeglasses:")

st.title(page_title)

st.markdown("""
Use the sidebar to set properties of your science target and tip-tilt star.

 1. Set the zenith angle of your science target in degrees.
 2. Set the V-band of your tip-tilt star (Current Assumptions: TT sensing at R-band and TT spectral type is K-type).
 3. Set the distance between the tip-tilt star and the science tareget, which is assumed to be at the center of the LGS constellation.
 4. Set the individual science exposure times. The PSF is blurred for longer exposures. 
 5. Click Calculate and your plots will appear below.
""")

results_cont = st.beta_container()

st.markdown("""
### About

The KAPA Strehl calculator was developed by the UC Berkeley Moving Universe Lab in collaboration with the
Keck All-Sky Precision Adaptive Optics (KAPA) Engineering and Science Tools Teams. Authors include:
Jessica Lu (UC Berkeley), Peter Wizinowich (WMKO), Matthew Freeman (UC Berkeley), Carlos Corriea (WMKO).
The code development and webpage is funded by the Gordon and Betty Moore Foundation and the National Science
Foundation. 

**Issues:** If you have found a bug or have a question, please submit an 
issue on our [GitHub repository](https://github.com/jluastro/kapa_strehl_calc/issues). 
""")

icon_col1, icon_col2, icon_col3 = st.beta_columns(3)

with icon_col1:
    st.image('figures/icon_mulab.png', width=100)

with icon_col2:
    st.image('figures/icon_ucsd_oir.png', width=100)

with icon_col3:
    st.image('figures/icon_keck.png', width=100)


##########
# Sidebar
##########    
st.sidebar.markdown("## Target Parameters")

# Input: Zenith Angle
in_za = st.sidebar.number_input('Zenith Angle (deg) -- range: [0 - 50]',
                            min_value=0, max_value=50, value=30, key='in_za')
# Input: Tip-Tilt V Mag
in_ttv = st.sidebar.number_input('TT Star V-band Brightness (mag) -- range: [9 - 19]',
                             min_value=9, max_value=20, value=10, key='in_ttv')
# Input: Tip-Tilt Offset
in_tto = st.sidebar.number_input('TT Star Offset (asec) -- range: [0 - 20]',
                             min_value=0, max_value=20, value=0, key='in_tto')
# Input: Science Exposure Integration Time
in_texp = st.sidebar.number_input('Science Exposure Time (sec) - range: [30, 900]',
                              min_value=30, max_value=900, value=30, key='in_texp')

# Input: Spectral Type of TT star - used for converting from V-band to TT WFS band
# add OBAFGKM in the future.
in_tt_type = st.sidebar.selectbox('TT Spectral Type', ['K'])

# Input: TT wavefront sensing band-pass -- add JH and K in the future
in_tt_band = st.sidebar.radio('TT Sensing Band', ['R']) 

####################
# Prep Data
####################

tab = load_data_table()

grid_inputs = prep_grid(tab)


####################
# Calculate
####################
if st.sidebar.button('Calculate', key='calc'):
    # perform the interpolation
    user_inputs = np.array([in_za, in_texp, in_tto, in_ttv])
    strehl, fwhm, ttm = calc_performance(user_inputs, grid_inputs)

    filts = np.array(['Y', 'Z', 'J', 'H', 'K'])

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(5, 6))
    plt.subplots_adjust(hspace=0.07, top=0.84)

    fmt1 = "TT: V={0:.1f} mag"
    fmt2 = 'TT Offset from Target: {0:.1f}"'
    fmt3 = 'Science Exposure Time: {0:.0f} sec'
    fmt4 = 'Zenith Angle: {0:.0f} deg'
    dy = 0.03
    
    plt.figtext(0.2, 0.95 - 0*dy, fmt4.format(in_za), fontsize=10)
    plt.figtext(0.2, 0.95 - 1*dy, fmt1.format(in_ttv), fontsize=10)
    plt.figtext(0.2, 0.95 - 2*dy, fmt2.format(in_tto), fontsize=10)
    plt.figtext(0.2, 0.95 - 3*dy, fmt3.format(in_texp), fontsize=10)

    ax1.plot(filts, ttm, 'b.--', ms=10)
    ax1.invert_yaxis()
    ax1.set_ylabel('Actual TT (mag)', fontsize=10)
    ax1.tick_params(axis='y', labelsize=10)

    ax2.plot(filts, strehl, 'b.--', ms=10)
    ax2.set_ylabel('Strehl', fontsize=10)
    ax2.tick_params(axis='y', labelsize=10)

    ax3.plot(filts, fwhm, 'b.--', ms=10)
    ax3.set_ylabel('FWHM (mas)', fontsize=10)
    ax3.set_xlabel('Filters', fontsize=10)
    ax3.tick_params(axis='y', labelsize=10)
    ax3.tick_params(axis='x', labelsize=10)

    f.align_ylabels()

    # Display the results
    results_cont.subheader('Results')

    results_cont.pyplot(f)

    # Make a data frame for display
    data = {'Filter': filts, 'TT Mag': ttm, 'Strehl': strehl, 'FWHM': fwhm}
    df = pandas.DataFrame(data=data)
    results_cont.dataframe(data)

    # -- Allow data download
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download Data as CSV File</a>'
    results_cont.markdown(href, unsafe_allow_html=True)
    
    
