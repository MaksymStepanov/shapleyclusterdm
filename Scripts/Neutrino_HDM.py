import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import os

def IceCubeDec30():
    data = [(6287.089059917242,2.176896784350992e-7),	
    (8071.918051887899,1.2826498305280624e-7),	
    (10000,7.796360130405253e-8),	
    (11534.944356766646,5.2025494423727083e-8),	
    (13789.067060798505,3.69460120519931e-8),	
    (18022.455152500002,2.1768967843509876e-8),	
    (29182.08237644098,6.468607661546321e-9),	
    (38828.20258008992,3.1622776601683795e-9),	
    (47251.82693586871,2.045553350050195e-9),	
    (64003.17969534017,1.0316051783820777e-9),	
    (85159.39318253836,5.892102187612295e-10),	
    (107400.85826829619,3.69460120519931e-10),	
    (135451.22770005654,2.2456979955397718e-10),	
    (177036.1740799287,1.3231882072236435e-10),	
    (219323.55227333188,1.0642092440647268e-10),	
    (286658.18124639796,8.296958520834915e-11),	
    (381412.8748283223,6.270429615031004e-11),	
    (564856.3098457617,4.45295850994266e-11),	
    (807191.8051887916,4.3165336925959006e-11),	
    (1195416.86845759,4.1842885079015755e-11),	
    (1802245.5152500002,4.3165336925959006e-11),	
    (2397976.712644727,4.45295850994266e-11),	
    (3488475.352951643,4.8886527451139877e-11),	
    (5959280.0958308065,5.5366012092767805e-11),	
    (10550081.484365547,6.883952069645511e-11),	
    (15624223.745758843,8.042765483057628e-11),	
    (22327351.48777292,9.396648314954671e-11),	
    (30242671.703095257,1.132541315152808e-10),	
    (53540468.92271295,1.4985653124037577e-10),	
    (93109125.58090752,2.045553350050195e-10),
    (156242237.45758876,2.792196291524867e-10),	
    (271711817.27004206,4.0560949048976497e-10),	
    (447881156.23460037,5.711586478126446e-10),
    (606660524.7569745,6.88395206964551e-10)]

    energy_GeV, flux_TeV = zip(*data)
    plt.loglog(energy_GeV, flux_TeV, ls='-', lw=2, alpha=1, color = 'brown', label= r'Ice Cube $\delta^{\circ} = -30^{\circ}$ 10 years')  


def plotKM3NETsens():
    dataSensitivityO = pd.read_csv('orange.csv', delimiter=';', header=None)
    dataSensitivityG = pd.read_csv('green.csv', delimiter=';', header=None)
    
    dataSensitivityO[0] = dataSensitivityO[0].str.replace(',', '.').astype(float)  # Energy (in GeV)
    dataSensitivityO[1] = dataSensitivityO[1].str.replace(',', '.').astype(float)  # Flux (in GeV cm^-2 s^-1)
    
    dataSensitivityG[0] = dataSensitivityG[0].str.replace(',', '.').astype(float)  # Energy (in GeV)
    dataSensitivityG[1] = dataSensitivityG[1].str.replace(',', '.').astype(float)  # Flux (in GeV cm^-2 s^-1)
    
    # Convert the flux from GeV/cm^2/s to TeV/cm^2/s
    fluxO_TeV = dataSensitivityO[0] / 1000 
    fluxG_TeV = dataSensitivityG[0] / 1000  
    
    
    plt.loglog(dataSensitivityO[1], fluxO_TeV, ls = '--', label=r'$\delta^{\circ} = -65^{\circ} 10 years$', color='black')
    plt.loglog(dataSensitivityG[1], fluxG_TeV, ls = '-', label=r'$\delta^{\circ} = -0.01^{\circ} 10 years$', color='black')

channel = "bb"
process = "ann"
mass = 10000
# Load data from file, skipping the first row (header)
filename = channel + '_' + process + '_' + 'nn' + '_' + f'{mass}'.format(mass = mass) + ".txt"  # Change this to your actual file name
data = np.loadtxt(filename, skiprows=1)  # Skips the first row

# Extract x and y columns from the file
x_raw = data[:, 0]  # Original x values in PeV
y_raw = data[:, 1]  # Original y values in PeV^-1 cm^-2 s^-1

# Convert dark matter mass from GeV to TeV
mass_GeV = 1e4 # Dark matter mass in GeV
mass_TeV = mass / 1000  # Convert mass to TeV

# Convert x to E in TeV

if process == "ann":
   E = (x_raw * mass_TeV)
else:
   E = (x_raw * mass_TeV)/2 


# Convert y to dN/dE in TeV^-1 cm^-2 s^-1
if process == "ann":
 dN_dE = (y_raw / mass_TeV)
else:
 dN_dE = 2*(y_raw / mass_TeV)

# Constants
#crosssection = [3 * 10**(-26), 3 * 10**(-22)]   # Cross-section in cm³/s

crosssection = [10**(25), 10**(27)]

# Cluster data (J-factors in GeV² cm⁻⁵)
clustersAnn = {
    "Perseus": {"j_factor": 2.5e+17, "redshift": 0.0179},
    "A3558": {"j_factor": 4.23e+16, "redshift": 0.048},
    "A3562": {"j_factor": 2.92e+16, "redshift": 0.049},
    "A3560": {"j_factor": 2.01e+16, "redshift": 0.0495},
    "A3556": {"j_factor": 1.26e+16, "redshift": 0.049}
}

clustersDec = {
   "Perseus": {"j_factor": 2.24e+19, "redshift": 0.0179},
   "A3558": {"j_factor": 3.78e+18, "redshift": 0.048},
    "A3562": {"j_factor": 2.45e+18 , "redshift": 0.049},
   "A3560": {"j_factor": 1.60e+18, "redshift": 0.0495},
   "A3556": {"j_factor": 9.26e+17, "redshift": 0.049}
}

# Convert J-factors from GeV² cm⁻⁵ to TeV² cm⁻⁵
for cluster in clustersAnn:
    clustersAnn[cluster]["j_factor"] /= 1e6  # Convert GeV² to TeV²

for cluster in clustersDec:
    clustersDec[cluster]["j_factor"] /= 1e3  # Convert GeV² to TeV²


# Create a single figure
plt.figure(figsize=(9, 6))


cluster_colors = {
        'Perseus': 'red',
        'A3558': 'blue',
        'A3562': 'green',
        'A3560': 'purple',
        'A3556': 'yellow'
    }

# Loop over clusters and plot all on the same figure
for cluster, values in clustersDec.items():
    for value in crosssection: 
     j_factor = values["j_factor"]
     redshift = values["redshift"]

     if(process == 'ann'):
      dmfactor = value / (8 * np.pi * mass_TeV**2)
     else:
      dmfactor = 1 / (4 * value * np.pi * mass_TeV)
    # Apply transformations
     y_transformed = dmfactor * dN_dE * j_factor * (E**2)
     x_transformed = (E / (1 + redshift))

     # Plot each cluster
     plt.plot(x_transformed, y_transformed, linestyle='-' if value == crosssection[0] else '--', color=cluster_colors[cluster], label=f"{cluster}" if value == crosssection[0] else None, markersize=2)

for value in crosssection:
              power = ""
              if (value == crosssection[0]): 
                x_position = 0.25
                y_position = 0.1
                power = 22
              else:
                x_position = 0.25
                y_position = 0.45
                power = 18
              plt.text(
                x_position,
                y_position,
                rf"$\sigma_{{ann}}v = 3 \times 10^{{-{power}}}$" if process == 'ann' else
                rf"$\tau_{{dec}} = 1 \times 10^{{{value}}}$",
                fontsize=10,
                color='black',
                ha='center',  # Horizontal alignment
                va='center',  # Vertical alignment
                transform=plt.gca().transAxes,  # Use axes coordinates (0,0 bottom-left, 1,1 top-right)
              )
              
IceCubeDec30()

plotKM3NETsens()

# Set logarithmic scales
plt.xscale("log")
plt.yscale("log")

    # Set axis limits
plt.xlim(1e-2 if process == 'Ann' else 1e-2, 1e+8)
if mass == 100000:
    plt.ylim(1e-22 if process == 'Ann' else 1e-17, 5e-15 if process == 'Ann' else 5e-12)
elif mass == 10000:
     plt.ylim(1e-22 if process == 'Ann' else 1e-17, 5e-15 if process == 'Ann' else 5e-12) 

# Format x-axis labels as powers of 10
plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=10.0))
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"$10^{{{int(np.log10(x))}}}$"))

plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"$10^{{{int(np.log10(y))}}}$"))

# Labels, title, and legend
plt.xlabel(r'$E$ [TeV]', fontsize=14)
plt.ylabel(r'$E^2 \times dN/dE$ [TeV cm$^{-2}$ s$^{-1}$]', fontsize=14)
plt.legend(loc='lower right', fontsize=12)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Grid lines only at powers of 10 (major ticks)
plt.grid(True, which="major", linestyle="--", linewidth=0.5)



if process == '_ann':
     if channel == 'bb' or channel == 'tt':
      plt.title(r"$\chi \chi \to$" + r"${0} \bar {0}$".format(channel[0]) + r"$\to \nu \bar \nu  $" + "," + " " + f"m = {mass_GeV/1000} TeV".format(mass_GeV = mass_GeV), fontsize=14)
     elif channel == 'zz' or channel == 'hh':
      channelUper = channel[0].upper() 
      plt.title(r"$\chi \chi \to$" + r"${0} {0}$".format(channelUper) + r"$\to \nu \bar \nu  $" + "," + " " + f"m = {mass_GeV/1000} TeV".format(mass_GeV = mass_GeV), fontsize=14)
     else:
      channelUper = channel[0].upper() 
      plt.title(r"$\chi \chi \to$" + r"${0}^+ {0}^-$".format(channelUper) + r"$\to \nu \bar \nu  $" + "," + " " + f"m = {mass_GeV/1000} TeV".format(mass_GeV = mass_GeV), fontsize=14)
elif process == '_Dec':
     channelUper = channel[0].upper() 
     if channel == 'bb' or channel == 'tt':
      plt.title(r"$\chi \to$" + r"${0} \bar {0}$".format(channel[0]) + r"$\to \nu \bar \nu  $" + "," + " " + f"m = {mass_GeV/1000} TeV".format(mass_GeV = mass_GeV), fontsize=14)
     elif channel == 'zz' or channel == 'hh':
      channelUper = channel[0].upper() 
      plt.title(r"$\chi \to$" + r"${0} {0}$".format(channelUper) + r"$\to \nu \bar \nu  $" + "," + " " + f"m = {mass_GeV/1000} TeV".format(mass_GeV = mass_GeV), fontsize=14)
     else:
      channelUper = channel[0].upper() 
      plt.title(r"$\chi \to$" + r"${0}^+ {0}^-$".format(channelUper ) + r"$\to \nu \bar \nu  $" + "," + " " + f"m = {mass_GeV/1000} TeV".format(mass_GeV = mass_GeV), fontsize=14)

output_dir = r"C:\Users\folder"  # Change this to your desired path

# Ensure the directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = os.path.join(output_dir, "ClusterN_" + f"{mass}" + "_" + f"{process}" + "_" + channel + ".png")

plt.savefig(output_file, dpi=300)

# Show the plot
plt.show()
