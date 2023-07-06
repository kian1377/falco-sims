import numpy as np
import copy
from astropy.io import fits
import matplotlib.pyplot as plt
plt.rcParams['image.origin']='lower'
import os
from pathlib import Path

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

import misc_funs as misc

def ipr(mp):# %% Perform an idealized phase retrieval (get the E-field directly)


    optval = copy.copy(mp.full)
    optval.source_x_offset = 0
    optval.use_dm1 = True
    optval.use_dm2 = True
    nout = 1024
    optval.output_dim = 1024
    optval.use_fpm = False
    optval.use_pupil_mask = False  # No SPM for getting initial phase
    optval.use_lyot_stop = False
    optval.use_field_stop = False
    optval.use_pupil_lens = True
    delattr(optval, 'final_sampling_lam0')

    # Use non-SPC flat maps for SPC since SPM has separate aberrations
    # downstream that can't be fully captured at entrance pupil with the SPM in
    # place. The SPM aberrations are flattened in a separate step not included
    # here.
    if 'sp' in mp.coro.lower():
        optval.dm1_m = mp.full.dm1.flatmapNoSPM
        optval.dm2_m = mp.full.dm2.flatmapNoSPM
    else:
        optval.dm1_m = mp.full.dm1.flatmap
        optval.dm2_m = mp.full.dm2.flatmap

    if mp.Nsbp == 1:
        lambdaFacs = np.array([1.])
    else:
        lambdaFacs = np.linspace(1-mp.fracBW/2, 1+mp.fracBW/2, mp.Nsbp)

    # Get the Input Pupil's E-field
    nCompact = falco.util.ceil_even(mp.P1.compact.Nbeam + 1)
    mp.P1.compact.E = np.ones((nCompact, nCompact, mp.Nsbp), dtype=complex)
    for iSubband in range(mp.Nsbp):

        lambda_um = 1e6*mp.lambda0*lambdaFacs[iSubband]

        # Get aberrations for the full optical train
        optval.pinhole_diam_m = 0  # 0 means don't use the pinhole at FPAM
        fieldFullAll, sampling = proper.prop_run('roman_phasec', lambda_um, nout, QUIET=True, PASSVALUE=optval.__dict__)
        misc.imshow2(np.abs(fieldFullAll), np.angle(fieldFullAll), pxscl=sampling)

        # Put pinhole at FPM to get back-end optical aberrations
        optval.pinhole_diam_m = mp.F3.pinhole_diam_m;
        fieldFullBackEnd, sampling = proper.prop_run('roman_phasec', lambda_um, nout, QUIET=True, PASSVALUE=optval.__dict__)
        optval.pinhole_diam_m = 0  # 0 means don't use the pinhole at FPAM
        misc.imshow2(np.abs(fieldFullBackEnd), np.angle(fieldFullBackEnd), pxscl=sampling)

        # Subtract off back-end phase aberrations from the phase retrieval estimate
        phFrontEnd = np.angle(fieldFullAll) - np.angle(fieldFullBackEnd)
        misc.imshow1(phFrontEnd, pxscl=sampling)

        # Put front-end E-field into compact model
        fieldFull = np.abs(fieldFullAll) * np.exp(1j*phFrontEnd)
        fieldCompactReal = falco.mask.rotate_shift_downsample_pupil_mask(np.real(fieldFull), mp.P1.full.Nbeam, mp.P1.compact.Nbeam, 0, 0, 0)
        fieldCompactImag = falco.mask.rotate_shift_downsample_pupil_mask(np.imag(fieldFull), mp.P1.full.Nbeam, mp.P1.compact.Nbeam, 0, 0, 0)
        fieldCompact = fieldCompactReal + 1j*fieldCompactImag
        fieldCompact = falco.util.pad_crop(fieldCompact, (nCompact, nCompact))
        mp.P1.compact.E[:, :, iSubband] = falco.prop.relay(fieldCompact, 1, centering=mp.centering)

        if mp.flagPlot:
            plt.figure(11); plt.imshow(np.angle(fieldCompact)); plt.colorbar(); plt.hsv(); plt.pause(1e-2)
            plt.figure(12); plt.imshow(np.abs(fieldCompact)); plt.colorbar(); plt.magma(); plt.pause(0.5)

    # Don't double count the pupil amplitude with the phase retrieval and a model-based mask
    mp.P1.compact.mask = np.ones_like(mp.P1.compact.mask)
    
    return mp



