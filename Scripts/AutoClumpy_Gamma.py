import os
import shutil
import time
import subprocess

# Define clusters with their J-factor and redshift values
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


channels = {
    "bb": 11,
    "tt": 12,
    "ww": 15,
    "zz": 18,
    "hh": 21
}

sigmav = 22

tau = 25

mass = 1000

dmprocess = "Dec"

output_folder = '/home/path/output'
max_iterations = 25
clumpy_output_file = f'spectra_CIRELLI11_EW_GAMMA_m{mass}.txt'  # This is the file CLUMPY generates



def get_br_string(position):
    br = [0] * 28
    br[position] = 1  # Set 1 at the required position (index is position-1)
    return ','.join(map(str, br))

def rename_existing_files(cluster, channel, DMprocess):
    """Renames the CLUMPY output file to spectra_{cluster}_Ann_{channel}.txt."""
    original_file_path = os.path.join(output_folder, clumpy_output_file)
    if os.path.exists(original_file_path):
        if dmprocess == "Ann":
           new_file_name = f"spectra_{cluster}_{DMprocess}{sigmav}_{channel}_m{mass}.txt"
        elif dmprocess == "Dec":
           new_file_name = f"spectra_{cluster}_{DMprocess}{tau}_{channel}_m{mass}.txt"
        new_file_path = os.path.join(output_folder, new_file_name)
        shutil.move(original_file_path, new_file_path)
        print(f"Renamed {clumpy_output_file} to {new_file_name}")

def check_for_file():
    """Check if the CLUMPY output file appears in the output folder."""
    return os.path.exists(os.path.join(output_folder, clumpy_output_file))

def run_clumpy_command(cluster, value, redshift, br_string, DMprocess):

    processvalue = None

    if DMprocess == "Ann":
        processvalue = 1
    elif DMprocess == "Dec":
        processvalue = 0

    """Runs the CLUMPY command."""
    command = (
        f'clumpy -z -i clumpy_params_main.txt --gPP_DM_ANNIHIL_SIGMAV_CM3PERS=3e-{sigmav} '
        f'--gPP_DM_IS_ANNIHIL_OR_DECAY={processvalue} --gPP_DM_DECAY_LIFETIME_S=1e+{tau} '
        f'--gPP_DM_MASS_GEV={mass} --gSIM_JFACTOR={value} --gSIM_REDSHIFT={redshift} '
        f'--gPP_BR={br_string}'
    )
    print(f"Running command: {command}")
    process = subprocess.Popen(command, shell=True)
    return process

def run_iterations():
    iteration_count = 0
    clusters = {}
    DMprocess = dmprocess

    if DMprocess == "Ann":
        clusters = clustersAnn
    elif DMprocess == "Dec":
        clusters = clustersDec
        
    for cluster, data in clusters.items():
        j_factor = data["j_factor"]
        redshift = data["redshift"]

        for channel_name, position in channels.items():
            if iteration_count >= max_iterations:
                print("Reached maximum iterations. Exiting.")
                return
            
            # Prepare the BR string for the current channel
            br_string = get_br_string(position)

            
            # Run CLUMPY command
            process = run_clumpy_command(cluster, j_factor, redshift, br_string, DMprocess)

            # Wait for the output file to appear in the output folder
            print(f"Waiting for {clumpy_output_file} to appear...")
            while not check_for_file():
                time.sleep(1)

            # Once the file is detected, stop CLUMPY and rename the file
            print(f"{clumpy_output_file} detected. Stopping process and renaming file.")
            process.terminate()

            # Rename the newly created file
            rename_existing_files(cluster, channel_name, DMprocess)

            iteration_count += 1

run_iterations()
