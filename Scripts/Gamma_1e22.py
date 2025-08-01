import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import os

def PlotOmazh2():
      # Energy log10(E/eV) values
    logE = np.array([
        18.45, 18.55, 18.65, 18.75, 18.85, 18.95, 19.05, 19.15, 19.25, 19.35,
        19.45, 19.55, 19.65, 19.75, 19.85, 19.95, 20.05, 20.15
    ])
    
    # Flux J and errors (unit: km^-2 yr^-1 sr^-1 eV^-1)
    flux = np.array([
        1.9383e-18, 9.076e-19, 4.310e-19, 2.164e-19, 1.227e-19, 6.852e-20,
        3.796e-20, 2.055e-20, 1.035e-20, 0.533e-20, 2.492e-21, 1.252e-21,
        5.98e-22, 1.93e-22, 8.10e-23, 1.86e-23, 5.5e-24, 2.9e-24
    ])

    # Symmetric error bars (take average of ± if asymmetric)
    flux_err_up = np.array([
        0.0067* 1e-18, 0.042* 1e-19, 0.025* 1e-19, 0.016* 1e-19, 0.011* 1e-19, 0.074* 1e-20, 0.049* 1e-20, 0.032* 1e-20,
        0.021* 1e-20, 0.013* 1e-20, 0.081* 1e-21, 0.052* 1e-21, 0.32* 1e-22, 0.17* 1e-22, 0.99* 1e-23, 0.46* 1e-23, 2.5* 1e-24, 1.7* 1e-24
    ])   # Convert error magnitude

    flux_err_down = np.array([
        0.0067* 1e-18, 0.041* 1e-19, 0.025* 1e-19, 0.016* 1e-19, 0.011* 1e-19, 0.073* 1e-20, 0.049* 1e-20, 0.032* 1e-20,
        0.02* 1e-20, 0.013* 1e-20, 0.079* 1e-21, 0.05* 1e-21, 0.31* 1e-22, 0.15* 1e-22, 0.88* 1e-23, 0.38* 1e-23, 1.8* 1e-24, 1.2* 1e-24
    ])  

    # Convert logE → E (in eV)
    E = 10**logE  # still in eV

    # Calculate E^3 * J(E) (in eV^2 km^-2 yr^-1 sr^-1)
    y = flux * E**3
    y_err_up = flux_err_up * E**3
    y_err_down = -flux_err_down * E**3

    y /= (1e10 * 365*24*60*60 * 1e24)
    y_err_up /= (1e10 * 365*24*60*60 * 1e24)
    y_err_down /= (1e10 * 365*24*60*60 * 1e24)
    E /= 1e12 

    # Plot
    plt.errorbar(E, y, yerr=y_err_up, yerr2 = y_err_down, fmt="o", color="black", label="Data from table")


# ------------------------------- Setup -----------------------------------
channel = "zz"
process = "dec"
mass = 10000000000000.0  # GeV
mass_GeV = 10e13
finalstate = 'gamma'
filename = f"{channel}_{process}_{finalstate}_1e22.txt"

# Load data
data = np.loadtxt(filename, skiprows=1)
x_raw = data[:, 0]
y_raw = data[:, 1]

# Convert units
mass_TeV = 1e10  # Convert GeV to TeV
E = (x_raw * mass_TeV) if process == "ann" else (x_raw * mass_TeV / 2)
dN_dE = (y_raw / mass_TeV) if process == "ann" else (2 * y_raw / mass_TeV)
if (process == 'ann'):
 crosssection = [3e-26, 3e-22]
else:
 crosssection = [10**(25), 10**(27)]
# Cluster data
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

for cluster in clustersAnn:
    clustersAnn[cluster]["j_factor"] /= 1e6  # Convert GeV² to TeV²

for cluster in clustersDec:
    clustersDec[cluster]["j_factor"] /= 1e3  





# ------------------------------ Plotting --------------------------------
plt.figure(figsize=(9, 6))
cluster_colors = {
    'Perseus': 'red',
    'A3558': 'blue',
    'A3562': 'green',
    'A3560': 'purple',
    'A3556': 'yellow'
}

output_dir = r"C:\Users\folder"
data_dir = os.path.join(output_dir, "Data")  # Folder for txt outputs
os.makedirs(data_dir, exist_ok=True)

for cluster, values in clustersAnn.items():
    for idx, value in enumerate(crosssection):
        j_factor = values["j_factor"]
        redshift = values["redshift"]

        if(process == 'ann'):
         dmfactor = value / (8 * np.pi * mass_TeV**2)
        else:
         dmfactor = 1 / (4 * value * np.pi * mass_TeV)
         
        y_transformed = dmfactor * dN_dE * j_factor * (E**3)

        # Divide by solid angle to get flux per sr
        
        theta_deg = 2.7
        theta_rad = np.radians(theta_deg)
        solid_angle_sr = 2 * np.pi * (1 - np.cos(theta_rad))  # example: 0.24 msr = 2.4e-4 sr
        y_transformed /= solid_angle_sr
        x_transformed = E / (1 + redshift)

        # Save to text file (only for the first cross-section to avoid overwriting)
        if idx == 0:
            output_txt = os.path.join(data_dir, f"{finalstate}_{cluster}_1e22_{channel}_{process}.txt")
            np.savetxt(output_txt, np.column_stack((x_transformed, y_transformed)),
                       header="Energy [TeV]    E^3 dN/dE [TeV^2 cm^-2 sr^-1 s^-1]", fmt="%.6e")

        label = f"{cluster}" if idx == 0 else None
        linestyle = '-' if idx == 0 else '--'
        color = cluster_colors[cluster]

        plt.plot(x_transformed, y_transformed, linestyle=linestyle, color=color, label=label)

# ----------------------------- Annotations ------------------------------
for idx, value in enumerate(crosssection):
    power = 27 if idx == 0 else 25
    x_pos = 0.9
    y_pos = 0.1 if idx == 0 else 0.45
    plt.text(x_pos, y_pos,
                             rf"$\sigma_{{ann}}v = 3 \times 10^{{-{power}}}$" if process == 'ann' else
                rf"$\tau_{{dec}} = 1 \times 10^{{{power}}}$",
             fontsize=10, color='black',
             ha='center', va='center',
             transform=plt.gca().transAxes)

#PlotOmazh2()

# ------------------------------- Plot -----------------------------------
plt.xscale("log")
plt.yscale("log")
if(process == 'ann'):
 plt.xlim(1e6, 1e11)
 plt.ylim(1e-32, 5e1)
else:
 plt.xlim(1e5, 1e11)
 plt.ylim(1e-23, 5e-16) 

plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=10.0))
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"$10^{{{int(np.log10(x))}}}$"))
plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"$10^{{{int(np.log10(y))}}}$"))

plt.xlabel(r'$E$ [TeV]', fontsize=14)
plt.ylabel(r'$E^3 \times dN/dE$ [TeV$^{2}$ cm$^{-2}$ sr$^{-1}$ s$^{-1}$]', fontsize=14)
plt.legend(loc='lower left', fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, which="major", linestyle="--", linewidth=0.5)

# ---------------------------- Title + Save ------------------------------
plt.title(r"$\chi \chi \to$" + r"${0} \bar {0}$".format(channel[0]) + r"$\to \gamma $" + "," + " " + r"m = $10^{10}$ TeV", fontsize=14)

if process == 'ann':
     if channel == 'bb' or channel == 'tt':
      plt.title(r"$\chi \chi \to$" + r"${0} \bar {0}$".format(channel[0]) + r"$\to \gamma $" + "," + " " + r"m = $10^{10}$ TeV", fontsize=14)
     elif channel == 'zz' or channel == 'hh':
      channelUper = channel[0].upper() 
      plt.title(r"$\chi \chi \to$" + r"${0} {0}$".format(channelUper) + r"$\to \gamma $" + "," + " " + r"m = $10^{10}$ TeV", fontsize=14)
     else:
      channelUper = channel[0].upper() 
      plt.title(r"$\chi \chi \to$" + r"${0}^+ {0}^-$".format(channelUper) + r"$\to \gamma $" + "," + " " + r"m = $10^{10}$ TeV", fontsize=14)
elif process == 'dec':
     channelUper = channel[0].upper() 
     if channel == 'bb' or channel == 'tt':
      plt.title(r"$\chi \to$" + r"${0} \bar {0}$".format(channel[0]) + r"$\to \gamma $" + "," + " " + r"m = $10^{10}$ TeV", fontsize=14)
     elif channel == 'zz' or channel == 'hh':
      channelUper = channel[0].upper() 
      plt.title(r"$\chi \to$" + r"${0} {0}$".format(channelUper) + r"$\to \gamma $" + "," + " " + r"m = $10^{10}$ TeV", fontsize=14)
     else:
      channelUper = channel[0].upper() 
      plt.title(r"$\chi \to$" + r"${0}^+ {0}^-$".format(channelUper ) + r"$\to \gamma $" + "," + " " + r"m = $10^{10}$ TeV", fontsize=14)

os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, f"Clusters_{finalstate}_{process}_1e22_{channel}.png"), dpi=300)
plt.show()


