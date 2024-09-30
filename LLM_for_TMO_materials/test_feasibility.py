import os
import subprocess
import pandas as pd
from mp_api.client import MPRester
from ase.io import read
from dotenv import load_dotenv
import sys
import argparse

load_dotenv()

MP_API_KEY = os.getenv("MP_API_KEY")

def identify_duplicates(cif_directory):
    for file in os.listdir(cif_directory):
        try:
            atoms = read(cif_directory + file,format = "cif")
            with MPRester(MP_API_KEY) as mpr:
                docs = mpr.materials.summary.search(formula=atoms.get_chemical_formula())
                print(atoms.get_chemical_formula())
                if docs:
                    print("Removing duplicate " + file)
                    os.system("rm " + cif_directory + file) 
        except:
            print("Error parsing CIF, likely invalid, deleting file")
            os.system("rm " + cif_directory + file) 



def predict_and_save_to_csv(cif_directory, output_csv):
    # List to store the results
    results = []

    for filename in os.listdir(cif_directory):
        if filename.endswith(".cif"):
            file_path = os.path.join(cif_directory, filename)
            try:
                # Read CIF content
                with open(file_path, 'r') as file:
                    cif_content = file.read()
               
                # Run the ALIGNN model to predict formation energy
                formation_energy_result = subprocess.check_output(
                    f"python ../CDVAE_for_TMO_materials/alignn/alignn/pretrained.py --model_name jv_formation_energy_peratom_alignn --file_format cif --file_path {file_path}",
                    shell=True
                )
                formation_energy = float(((str(formation_energy_result).split("[")[2]).split("]")[0]))
                print(f"Formation energy for {filename}: {formation_energy} eV/atom")

                # Run the ALIGNN model to predict energy above hull
                energy_above_hull_result = subprocess.check_output(
                    f"python ../CDVAE_for_TMO_materials/alignn/alignn/pretrained.py --model_name jv_ehull_alignn --file_format cif --file_path {file_path}",
                    shell=True
                )
                energy_above_hull = float(((str(energy_above_hull_result).split("[")[2]).split("]")[0]))
                print(f"Energy above hull for {filename}: {energy_above_hull} eV/atom")

                # Run the ALIGNN model to predict band gap
                band_gap_result = subprocess.check_output(
                    f"python ../CDVAE_for_TMO_materials/alignn/alignn/pretrained.py --model_name jv_mbj_bandgap_alignn --file_format cif --file_path {file_path}",
                    shell=True
                )
                band_gap = float(((str(band_gap_result).split("[")[2]).split("]")[0]))
                print(f"Band gap for {filename}: {energy_above_hull} eV/atom")

                # Append the results as a dictionary
                results.append({
                    "Name": filename,
                    "CIF": cif_content,
                    "Formation Energy (eV/atom)": formation_energy,
                   
                    "Energy Above Hull (eV/atom)": energy_above_hull,

                    "Band Gap (eV/atom)": band_gap,
                })

            except subprocess.CalledProcessError as e:
                print(f"Error processing {filename}: {e}")
            except ValueError as ve:
                print(f"Error parsing values for {filename}: {ve}")

    # Create a DataFrame from the results
    df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)

    print(f"Results have been saved to {output_csv}")

def main(args):
    # Specify the directory containing the CIF files and the output CSV file
    cif_directory = args.output_dir
    output_csv = sys.path[0] + "output.csv"

    identify_duplicates(cif_directory=cif_directory)

    predict_and_save_to_csv(cif_directory, output_csv)

    df = pd.read_csv(sys.path[0] + "output.csv")

    print(df[(df['Formation Energy (eV/atom)'] < -1.5) &
                 (df['Energy Above Hull (eV/atom)'] < 0.08) & 
                 (df['Band Gap (eV/atom)'] <= 0.5)])

    filtered_df = df[(df['Formation Energy (eV/atom)'] < -1.5) &
                 (df['Energy Above Hull (eV/atom)'] < 0.08) & 
                 (df['Band Gap (eV/atom)'] <= 0.5)]
    
    output_path = sys.path[0] + 'filtered_structures_unconditional.csv'
    filtered_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    main(args=args)

