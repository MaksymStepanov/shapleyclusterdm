from __future__ import print_function
# Load HDMSpectra
from HDMSpectra import HDMSpectra

# Import numpy
import numpy as np

# Plotting defaults
import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif' 
mpl.rcParams['xtick.labelsize'] = 26
mpl.rcParams['ytick.labelsize'] = 26
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.minor.size'] = 2.5
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['xtick.minor.width'] = 0.75
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.minor.size'] = 2.5
mpl.rcParams['ytick.major.width'] = 1.0
mpl.rcParams['ytick.minor.width'] = 0.75
mpl.rcParams['ytick.major.pad'] = 8
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['legend.fontsize'] = 26
mpl.rcParams['legend.frameon'] = False

#1, 2, 3, 4, 5, 6,       # Q
#11, 12, 13, 14, 15, 16, # L
#21, 22, 23, 24,         # V
#25                      # H

#'d', 'u', 's', 'c', 'b', 't',             # Q
#'e', 'nue', 'mu', 'numu', 'tau', 'nutau', # L
#'g', 'gamma', 'Z', 'W',                   # V
#'h'                                       # H

#11, -11,                   # electron
#12, -12, 14, -14, 16, -16, # neutrino
#22,                        # photon
#2212, -2212                # proton

#'e', 'ae',                                         # electron
#'nue', 'anue', 'numu', 'anumu', 'nutau', 'anutau', # neutrino
#'gamma',                                           # photon
#'p', 'ap'                                          # proton

# We'll leave the basic arguments as above

finalstate = 14
initialstate = 5
mDM = 1e13
mDM_TeV = 1e10
x = np.logspace(-6.,0,1000)
#annihilation=True
canal = 'Z'
process = 'dec'

# Extract both spectra
#dNdx_dec = HDMSpectra.spec(finalstate, initialstate, x, mDM, './data/HDMSpectra.hdf5', annihilation = False)

if process == 'ann': 
 dNdx = HDMSpectra.spec(finalstate, canal, x, mDM, './data/HDMSpectra.hdf5', annihilation = True)
else:
 dNdx = HDMSpectra.spec(finalstate, canal, x, mDM, './data/HDMSpectra.hdf5', annihilation = False)

# Compute y values
y = dNdx

# Stack x and y into two columns
data = np.column_stack((x, y))

# Save to file
np.savetxt(canal.lower() + canal.lower() + '_' + process + '_' + 'numu' + '_' + '1e22' + ".txt", data, fmt='%.8e', header='x    y')

# Plot the spectrum
fig, ax = plt.subplots(1, figsize=(10/1.1, 8/1.1))

#plt.plot(x, x**2.*dNdx_dec, 
         #lw=2.5, c='orange', label=r'Decay $m_{\chi} = 10^9$ GeV')

plt.plot(x, y, 
         lw=2.5, c='black',ls=':', label=r'Annihilation $m_{\chi} = 5 \times 10^8$ GeV')

plt.xscale('log')
plt.yscale('log')
plt.xlim([1.e-6,1.])
plt.ylim([1.e-6,2.e+9])
plt.xlabel(r"$x = 2E/m_{\chi}$", fontsize=30)
plt.ylabel(r"$x^2 dN/dx$", fontsize=30)
plt.legend()
plt.tight_layout()
plt.show()