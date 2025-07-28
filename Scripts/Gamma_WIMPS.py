import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define clusters, channels, masses, and values
clusters = ['Perseus', 'A3558', 'A3562', 'A3560', 'A3556']
channel = 'zz'
process = 'Ann'
places = ['South', 'North']
masses = [10000]

if process == 'Ann':
 values = ['22', '26']
elif process == 'Dec':
 values = ['25', '27']   

# Define output directory for graphs
output_dir = r"C:\Users\folder"  # Change this to your desired path

# Ensure the directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to construct input file path for a given cluster and channel
def get_input_path(cluster, value, channel, process, mass):
    return r"C:\Users\folder\{cluster}\spectra_{cluster}_{process}{value}_{channel}_m{mass}.txt".format(
        cluster=cluster, value=value, channel=channel, process=process, mass=mass
    )

# Function to construct CTA sensitivity file path
def get_input_pathCTA(place):
    return r"C:\Users\folder\CTA\CTA-{place}-20deg-50h-Sens.txt".format(place=place)

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

# Function to load CTA sensitivity data
def load_CTA_data():
    data = {}
    for place in places:
        input_path = get_input_pathCTA(place)
        if os.path.exists(input_path):
            data[place] = pd.read_csv(input_path, delimiter=r'\s+')
            print(f"Loaded data from {input_path}")
        else:
            print(f"File {input_path} not found. Skipping place {place}.")
    return data

# Function to plot data
def plot_data(cluster_data, CTA_data, mass):
    dataSouth = CTA_data.get('South')
    dataNorth = CTA_data.get('North')

    # Process CTA data
    if dataSouth is not None:
        dataSouth['E'] = dataSouth['E']
        dataSouth['E^{2}timesFlux'] = dataSouth['E^{2}timesFlux'] * 0.625

    if dataNorth is not None:
        dataNorth['E'] = dataNorth['E']
        dataNorth['E^{2}timesFlux'] = dataNorth['E^{2}timesFlux'] * 0.625

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
            if mass == 1000:
              y_offset = max(combined_dataset['E^{2}timesFlux']) * 0.001
              x_offset = max(combined_dataset['E']) * 0.02
              x_position += x_offset
              y_position -= y_offset
            elif mass == 10000:
              y_offset = max(combined_dataset['E^{2}timesFlux']) * 0.04
              y_position -= y_offset


            if process == 'Ann':
             plt.text(
                x_position,
                y_position,
                rf"$\sigma_{{ann}}v = 3 \times 10^{{-{value}}}$",
                fontsize=12,
                color='black',
                ha='center'
            )
            elif process == 'Dec':
              plt.text(
                x_position,
                y_position,
                rf"$\tau_{{dec}} = 1 \times 10^{{{value}}}$",
                fontsize=12,
                color='black',
                ha='center'
            )

    # Plot CTA sensitivity data
    if dataNorth is not None:
        plt.plot(
            dataNorth['E'],
            dataNorth['E^{2}timesFlux'],
            label='CTA North Sensitivity',
            linestyle='-',
            color='black',
            markersize=1
        )

    if dataSouth is not None:
        plt.plot(
            dataSouth['E'],
            dataSouth['E^{2}timesFlux'],
            label='CTA South Sensitivity',
            linestyle='--',
            color='black',
            markersize=1
        )

    # Set axis limits
    plt.xlim(1e-2, 1e+1)
    plt.ylim(1e-22 if process == 'Ann' else 1e-17, 1e-10 if process == 'Ann' else 1e-10)
 

    # Logarithmic scales
    plt.xscale('log')
    plt.yscale('log')

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)


    # Axis labels and legend
    plt.xlabel(r'$E$ [TeV]', fontsize=18)
    plt.ylabel(r'$E^2 \times Flux$ [TeV cm$^{-2}$ s$^{-1}$]', fontsize=18)
    plt.legend(loc='lower left', fontsize=14)

    if process == 'Ann':
     
     if channel == 'bb' or channel == 'tt':
      plt.title(r"$\chi \chi \to$" + r"${0} \bar {0}$".format(channel[0]) + r"$\to \gamma \gamma  $" + "," + " " + f"m = {mass/1000} TeV".format(mass = mass), fontsize=14)
     elif channel == 'zz' or channel == 'hh':
      channelUper = channel[0].upper() 
      plt.title(r"$\chi \chi \to$" + r"${0} {0}$".format(channelUper) + r"$\to \gamma \gamma  $" + "," + " " + f"m = {mass/1000} TeV".format(mass = mass), fontsize=14)
     else:
      channelUper = channel[0].upper() 
      plt.title(r"$\chi \chi \to$" + r"${0}^+ {0}^-$".format(channelUper) + r"$\to \gamma \gamma  $" + "," + " " + f"m = {mass/1000} TeV".format(mass = mass), fontsize=14)
    elif process == 'Dec':
     channelUper = channel[0].upper() 
     if channel == 'bb' or channel == 'tt':
      plt.title(r"$\chi \to$" + r"${0} \bar {0}$".format(channel[0]) + r"$\to \gamma \gamma  $" + "," + " " + f"m = {mass/1000} TeV".format(mass = mass), fontsize=14)
     elif channel == 'zz' or channel == 'hh':
      channelUper = channel[0].upper() 
      plt.title(r"$\chi \to$" + r"${0} {0}$".format(channelUper) + r"$\to \gamma \gamma  $" + "," + " " + f"m = {mass/1000} TeV".format(mass = mass), fontsize=14)
     else:
      channelUper = channel[0].upper() 
      plt.title(r"$\chi \to$" + r"${0}^+ {0}^-$".format(channelUper ) + r"$\to \gamma \gamma  $" + "," + " " + f"m = {mass/1000} TeV".format(mass = mass), fontsize=14)
    plt.grid(True, which="major", linestyle='--', linewidth=0.5)

    # Save the plot
    output_file = os.path.join(output_dir, f"Cluster_CTA_{process}_{channel}_mass{mass}_Test.png")
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.show()

# Main script
if __name__ == "__main__":
    CTA_data = load_CTA_data()
    for mass in masses:
            cluster_data = load_cluster_data(channel, mass)
            plot_data(cluster_data, CTA_data, mass)
