import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator, FuncFormatter
import numpy as np
import math
import os

# Define clusters and channels
clusters = ['Perseus','A3558', 'A3562', 'A3560', 'A3556']
factors = ['D', 'J']
process = 'Dec'
# Define the directory where you want to save the images
output_dir = r""  # Change this to your desired path

# Ensure the directory exists, create it if it doesn't
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # This creates the directory if it doesn't exist


# Function to construct input file path for a given cluster and channel
def get_input_path(cluster, factor):
    return r"C:\Users\folder\{cluster}\{cluster}_{factor}Factors.txt.txt".format(cluster=cluster, factor = factor)

# Function to load data for a given cluster
def load_cluster_data(factor):
    data = {}
    for cluster in clusters:
        input_path = get_input_path(cluster, factor)
        if os.path.exists(input_path):
            data[cluster] = pd.read_csv(input_path, delimiter=r'\s+', names=['alpha_int', 'F_alpha_int', 'D_alpha_int_normed', 'obj_name'], header=0)
            print(f"Loaded data from {input_path}")
        else:
            print(f"File {input_path} not found. Skipping cluster {cluster}.")

    return data

# User chooses a cluster
 #print("Available clusters:", clusters)
#selected_cluster = input("Enter the cluster you want to process (e.g., A3558): ")

for factor in factors:
 cluster_data = load_cluster_data(factor)

    # Now you can access the loaded data for each channel
 dataPer = cluster_data.get('Perseus')
 data58 = cluster_data.get('A3558')
 data62 = cluster_data.get('A3562')
 data60 = cluster_data.get('A3560')
 data56 = cluster_data.get('A3556')

    # Example: Access and work with data from the 'tt' channel
 if dataPer is not None:
       print("First few rows of the 'tt' channel data:")
       print(dataPer.head())
 else:
       print(f"Cluster {'Perseus'} is not available. Please choose a valid cluster.")


# Print the DataFrame to check the data
#print(data)

# Extract x and y values
 xPer = dataPer['alpha_int']
 yPer = dataPer['F_alpha_int']

 x58 = data58['alpha_int']
 y58 = data58['F_alpha_int']

 x62 = data62['alpha_int']
 y62 = data62['F_alpha_int']

 x60 = data60['alpha_int']
 y60 = data60['F_alpha_int']
 
 x56 = data56['alpha_int']
 y56 = data56['F_alpha_int']

# Create a scatter plot
 plt.plot(xPer, yPer, label=r'Perseus', marker='o', linestyle='-', color='red', markersize=0.5)
 plt.plot(x58, y58, label=r'A3558',marker='o', linestyle='-', color='blue', markersize=0.5)
 plt.plot(x62, y62, label=r'A3562',marker='o', linestyle='-', color='green', markersize=0.5)
 plt.plot(x60, y60, label=r'A3560',marker='o', linestyle='-', color='purple', markersize=0.5)
 plt.plot(x56, y56, label=r'A3556',marker='o', linestyle='-', color='yellow', markersize=0.5)


 plt.xlabel(r'$α_{int}$ [deg]', fontsize=20)
 if(factor == 'D'):
   plt.ylabel(r'$J_{dec}$ [GeV $cm^{-2}$]', fontsize=20)
 elif(factor == 'J'):
   plt.ylabel(r'$J_{ann}$ [$GeV^{2}$ $cm^{-5}$]', fontsize=20)


 plt.xlim(1e-2, 2.8e+0)  # Limit x-axis 
 #plt.ylim(1e-15, 1e-3)  # Limit y-axis 


 plt.grid(True, which="major", ls="--")

 plt.legend(loc='upper left', fontsize=13)

# Define a custom formatter for the log scale
 def log_formatx(x, pos):
    if x == 0:
        return '$0$'
    exponent = int(math.log10(x))
    return rf'$10^{{{exponent}}}$'
 
 def log_formaty(y, pos):
    if y == 0:
        return '$0$'
    exponent = int(math.log10(y))
    return rf'$10^{{{exponent}}}$'
 
 plt.xscale('log')
 plt.yscale('log')
# Format the x-axis in terms of 10^n
 ax = plt.gca()

# Set major and minor locators for x-axis
 ax.xaxis.set_major_locator(LogLocator(base=10.0))
 ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
 ax.xaxis.set_major_formatter(FuncFormatter(log_formatx))

# Set major and minor locators for y-axis
 ax.yaxis.set_major_locator(LogLocator(base=10.0))
 ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
 ax.yaxis.set_major_formatter(FuncFormatter(log_formaty))

# Increase the font size of tick labels
 ax.tick_params(axis='both', which='major', labelsize=19)
 ax.tick_params(axis='both', which='minor', labelsize=19)

    # Save the plot as a PNG file in the output directory
 filename = f"{factor}_factors.png"
 filepath = os.path.join(output_dir, filename)  # Combine directory with filename
 plt.savefig(filepath, dpi=300, bbox_inches='tight')
 print(f"Saved {filepath}")

 print("xPer range:", xPer.min(), xPer.max())
 print("yPer range:", yPer.min(), yPer.max())

# Clear the current plot before creating the next one
 plt.clf()

# Show the plot
 plt.show()