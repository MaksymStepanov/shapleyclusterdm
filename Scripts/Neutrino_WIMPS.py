import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
    
    
    plt.loglog(dataSensitivityO[1], fluxO_TeV, ls = '-', label=r'$\delta^{\circ} = -65^{\circ} 10 years$', color='black')
    plt.loglog(dataSensitivityG[1], fluxG_TeV, ls = '--', label=r'$\delta^{\circ} = -0.01^{\circ} 10 years$', color='black')

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
    energy_TeV = [e / 1000 for e in energy_GeV]
    plt.loglog(energy_TeV , flux_TeV, ls='-', lw=2, alpha=1, color = 'brown', label= r'Ice Cube $\delta^{\circ} = -30^{\circ}$ 10 years')  



# Define clusters, channels, masses, and values
clusters = ['Perseus', 'A3558', 'A3562', 'A3560', 'A3556']
channel = 'hh'
process = 'Dec'
places = ['South', 'North']
masses = [100000]

if process == 'Ann':
 values = ['18', '26']
elif process == 'Dec':
 values = ['25', '27']   

# Define output directory for graphs
output_dir = r"C:\Users\folder"  # Change this to your desired path

# Ensure the directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to construct input file path for a given cluster and channel
def get_input_path(cluster, value, channel, process, mass):
    return r"C:\Users\folder\{cluster}\spectraN_{cluster}_{process}{value}_{channel}_m{mass}.txt".format(
        cluster=cluster, value=value, channel=channel, process=process, mass=mass
    )

# Function to load cluster data
def load_cluster_data(channel, mass):
    data = {}
    for value in values:
        for cluster in clusters:
            input_path = get_input_path(cluster, value, channel, process, mass)
            if os.path.exists(input_path):
                data[cluster, value] = pd.read_csv(input_path, delimiter=r'\s+')
                print(f"Loaded data from {input_path}")
            else:
                print(f"File {input_path} not found. Skipping cluster {cluster}.")
    return data

# Function to plot data
def plot_data(cluster_data, mass):

    plt.figure(figsize=(10, 7))

    # Define specific colors for clusters
    cluster_colors = {
        'Perseus': 'red',
        'A3558': 'blue',
        'A3562': 'green',
        'A3560': 'purple',
        'A3556': 'yellow'
    }

    # Plot cluster data
    for cluster in clusters:
        for value in values:
            dataset = cluster_data.get((cluster, value))
            if dataset is not None:
                dataset['E'] = dataset['E'] / 10**6  # Convert to TeV
                dataset['E^{2}timesFlux'] = dataset['E^{2}timesFlux'] / 10**6  # Scale Flux
                plt.plot(
                    dataset['E'],
                    dataset['E^{2}timesFlux'],
                    linestyle='-' if value == values[0] else '--',
                    color=cluster_colors[cluster],
                    label=cluster if value == values[0] else None,  # Add cluster to legend only once
                    markersize=2
                )

    # Add single label for each value near the group's center
    for value in values:
        all_datasets = [
            cluster_data.get((cluster, value)) for cluster in clusters if cluster_data.get((cluster, value)) is not None
        ]
        if all_datasets:
            combined_dataset = pd.concat(all_datasets).sort_values(by='E')
            # Determine a safe position for the label
            middle_idx = len(combined_dataset['E']) // 2
            x_position = combined_dataset['E'].iloc[middle_idx]
            y_position = combined_dataset['E^{2}timesFlux'].iloc[middle_idx]

            # Shift the label position slightly downward to avoid crossing lines
            if mass == 10000:
              y_offset = max(combined_dataset['E^{2}timesFlux']) * -0.04
              x_offset = max(combined_dataset['E']) * 0.3
              x_position += x_offset
              y_position -= y_offset
            elif mass == 100000:
              y_offset = max(combined_dataset['E^{2}timesFlux']) * -0.01
              x_offset = max(combined_dataset['E']) * 3
              x_position += x_offset
              y_position -= y_offset

            if process == 'Ann':
             plt.text(
                x_position,
                y_position,
                rf"$\sigma_{{ann}}v = 3 \times 10^{{-{value}}}$",
                fontsize=10,
                color='black',
                ha='center'
            )
            elif process == 'Dec':
              plt.text(
                x_position,
                y_position,
                rf"$\tau_{{dec}} = 1 \times 10^{{{value}}}$",
                fontsize=10,
                color='black',
                ha='center'
            )

    # Plot IceCube and KM3NET sensitivity data
        
    IceCubeDec30()

    plotKM3NETsens()

    # Set axis limits
    plt.xlim(1e-2 if process == 'Ann' else 1e-2, 1e+7)
    if mass == 100000:
     plt.ylim(1e-22 if process == 'Ann' else 1e-17, 5e-9 if process == 'Ann' else 5e-10)
    elif mass == 10000:
     plt.ylim(1e-22 if process == 'Ann' else 1e-17, 5e-9 if process == 'Ann' else 5e-10) 

    # Logarithmic scales
    plt.xscale('log')
    plt.yscale('log')

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

  

    # Axis labels and legend
    plt.xlabel(r'$E$ [TeV]', fontsize=14)
    plt.ylabel(r'$E^2 \times Flux$ [TeV cm$^{-2}$ s$^{-1}$]', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)

   
    if process == 'Ann':
     
     if channel == 'bb' or channel == 'tt':
      plt.title(r"$\chi \chi \to$" + r"${0} \bar {0}$".format(channel[0]) + r"$\to \nu \bar \nu  $" + "," + " " + f"m = {mass/1000} TeV".format(mass = mass), fontsize=14)
     elif channel == 'zz' or channel == 'hh':
      channelUper = channel[0].upper() 
      plt.title(r"$\chi \chi \to$" + r"${0} {0}$".format(channelUper) + r"$\to \nu \bar \nu  $" + "," + " " + f"m = {mass/1000} TeV".format(mass = mass), fontsize=14)
     else:
      channelUper = channel[0].upper() 
      plt.title(r"$\chi \chi \to$" + r"${0}^+ {0}^-$".format(channelUper) + r"$\to \nu \bar \nu  $" + "," + " " + f"m = {mass/1000} TeV".format(mass = mass), fontsize=14)
    elif process == 'Dec':
     channelUper = channel[0].upper() 
     if channel == 'bb' or channel == 'tt':
      plt.title(r"$\chi \to$" + r"${0} \bar {0}$".format(channel[0]) + r"$\to \nu \bar \nu  $" + "," + " " + f"m = {mass/1000} TeV".format(mass = mass), fontsize=14)
     elif channel == 'zz' or channel == 'hh':
      channelUper = channel[0].upper() 
      plt.title(r"$\chi \to$" + r"${0} {0}$".format(channelUper) + r"$\to \nu \bar \nu  $" + "," + " " + f"m = {mass/1000} TeV".format(mass = mass), fontsize=14)
     else:
      channelUper = channel[0].upper() 
      plt.title(r"$\chi \to$" + r"${0}^+ {0}^-$".format(channelUper ) + r"$\to \nu \bar \nu  $" + "," + " " + f"m = {mass/1000} TeV".format(mass = mass), fontsize=14)
    plt.grid(True, which="major", linestyle='--', linewidth=0.5)
    plt.grid(True, which="major", linestyle='--', linewidth=0.5)

    # Save the plot
    output_file = os.path.join(output_dir, f"ClusterN_CTA_{process}_{channel}_mass{mass}.png")
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.show()

# Main script
if __name__ == "__main__":
    for mass in masses:
            cluster_data = load_cluster_data(channel, mass)
            plot_data(cluster_data, mass)
            
