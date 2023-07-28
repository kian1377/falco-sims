import numpy as np
import copy
from astropy.io import fits
from importlib import reload
import matplotlib.pyplot as plt
plt.rcParams['image.origin']='lower'
import os
from pathlib import Path
from datetime import datetime
date = int(datetime.today().strftime('%Y%m%d'))

import proper
proper.use_ffti = False
proper.use_fftw = False

import roman_phasec_proper as phasec
phasec.copy_here()
import falco

from IPython.display import clear_output, display, HTML
display(HTML("<style>.container { width:90% !important; }</style>")) # just making the notebook cells wider

falco_dir = Path(os.path.dirname(falco.__file__))
roman_dir = Path(os.path.dirname(falco_dir))/'roman'
flatmap_dir = roman_dir/'flatmaps'

import misc
from imshows import *

import falco_hlc_575_mp
import phase_retrieval as pr

# Initialize the model parameters
reload(falco_hlc_575_mp)
mp = falco_hlc_575_mp.setup_mp(Nitr=30, estimator='pwp-bp')
mp.runLabel = f'hlc_575_{mp.Nitr}itr_{mp.estimator}'
print(mp.runLabel)

# Set the initial DM state
dm1_flat_fpath = Path('~/roman_phasec_v1.4/python/roman_phasec_proper/examples/hlc_flattened_dm1.fits')
dm2_flat_fpath = Path('~/roman_phasec_v1.4/python/roman_phasec_proper/examples/hlc_flattened_dm2.fits')

dm1_flat = fits.getdata(dm1_flat_fpath)
dm2_flat = fits.getdata(dm2_flat_fpath)
imshow2(dm1_flat, dm2_flat)

mp.full.dm1 = falco.config.Object()
mp.full.dm2 = falco.config.Object()

mp.full.dm1.flatmap = dm1_flat
mp.full.dm2.flatmap = dm2_flat

mp.dm1.biasMap = 50 + mp.full.dm1.flatmap/mp.dm1.VtoH  # Bias voltage. Needed prior to WFSC to allow + and - voltages. Total voltage is mp.dm1.biasMap + mp.dm1.V
mp.dm2.biasMap = 50 + mp.full.dm2.flatmap/mp.dm2.VtoH  # Bias voltage. Needed prior to WFSC to allow + and - voltages. Total voltage is mp.dm2.biasMap + mp.dm2.V
imshow2(mp.dm1.biasMap, mp.dm2.biasMap)

# Run idealized PR
mp = pr.ipr(mp)

# Flesh out workspace
mp.TrialNum = 1
mp.SeriesNum = 1
out = falco.setup.flesh_out_workspace(mp)

# Run EFC
falco.wfsc.loop(mp, out)
misc.save_pickle('results/' + mp.runLabel + f'_mp_{date}.pkl', mp)
misc.save_pickle('results/' + mp.runLabel + f'_out_{date}.pkl', out)



