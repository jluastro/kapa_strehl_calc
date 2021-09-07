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
import calculations as calc

all_ref_filters = ['V', 'r', 'i', 'J', 'H', 'K']
all_spec_types = ['K']


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
 2. Set the individual science exposure times. The PSF is blurred for longer exposures. 
 3. Set the magnitude of your tip-tilt star in a reference filter.
 4. Set the the reference filter for your tip-tilt star magnitude (Choices: V, r, i, J, H, K).
 5. Set the distance between the tip-tilt star and the science tareget, which is assumed to be at the center of the LGS constellation.
 6. Click Calculate and your plots will appear below.

NOTE: If the plots are empty, you may be requesting values outside the currently available grid. Please alert Jessica Lu (jlu.astro@berkeley.edu) and include the desired values.
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

# Input: Science Exposure Integration Time
in_texp = st.sidebar.number_input('Science Exposure Time (sec) - range: [30, 900]',
                              min_value=30, max_value=900, value=30, key='in_texp')

# Input: Tip-Tilt Mag in Reference Filter
in_tt_ref_mag = st.sidebar.number_input('TT Brightness in Reference Filter (mag)',
                            min_value=6, max_value=20, value=10, key='in_tt_ref_mag')

# Input: TT Reference Band
in_tt_ref_filt = st.sidebar.selectbox('TT Reference Filter', all_ref_filters,
                                          key='in_tt_ref_filt')

# Input: Spectral Type of TT star - used for converting from V-band to TT WFS band
# add OBAFGKM in the future.
in_tt_type = st.sidebar.selectbox('TT Spectral Type', ['K'], key='in_tt_ref_type')

# Input: TT wavefront sensing band-pass -- add J,H and K in the future
in_tt_band = st.sidebar.radio('TT Sensing Band', ['R']) 

# Input: Tip-Tilt Offset
in_tto = st.sidebar.number_input('TT Star Offset (asec) -- range: [0 - 60]',
                             min_value=0, max_value=60, value=0, key='in_tto')


####################
# Prep Data
####################

tab = calc.load_data_tables()

grid_inputs = calc.prep_grid(tab)


####################
# Calculate
####################
if st.sidebar.button('Calculate', key='calc'):
    # perform the interpolation
    user_inputs = np.array([in_za, in_texp, in_tto, in_tt_ref_mag, in_tt_ref_filt, in_tt_type, in_tt_band])
    strehl, fwhm, ttm = calc.calc_performance(user_inputs, grid_inputs)

    filts = np.array(['Y', 'Z', 'J', 'H', 'K'])

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(5, 6))
    plt.subplots_adjust(hspace=0.07, top=0.84)

    fmt1 = 'Science Exposure Time: {0:.0f} sec'
    fmt2 = 'Zenith Angle: {0:.0f} deg'
    fmt3 = "TT: {0:s} = {1:.1f} mag (spec-type = {2:s})"
    fmt4 = 'TT Offset from Target: {0:.1f}"'
    dy = 0.03
    
    plt.figtext(0.2, 0.95 - 0*dy, fmt1.format(in_za), fontsize=10)
    plt.figtext(0.2, 0.95 - 1*dy, fmt2.format(in_texp), fontsize=10)
    plt.figtext(0.2, 0.95 - 2*dy, fmt3.format(in_tt_ref_filt, in_tt_ref_mag, in_tt_type), fontsize=10)
    plt.figtext(0.2, 0.95 - 3*dy, fmt4.format(in_tto), fontsize=10)

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
    
    
