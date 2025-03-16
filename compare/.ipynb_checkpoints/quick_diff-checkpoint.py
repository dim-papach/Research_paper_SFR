import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# Create a directory for saving plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# File paths for different datasets
outer_join_file = "/home/dp/Documents/Research_paper_SFR/tables/outer_join.csv"
inner_join_file = "/home/dp/Documents/Research_paper_SFR/tables/inner_join.csv"
unique_file1 = "/home/dp/Documents/Research_paper_SFR/tables/HEC_not_LVG_join.csv"
unique_file2 = "/home/dp/Documents/Research_paper_SFR/tables/LVG_not_HEC_join.csv"

# Path to the CSV defining x-y combinations and descriptions
combinations_file = "Reordered_Final_Comparison_Table.csv"  # Update with actual path

# Read the data
outer_join_data = pd.read_csv(outer_join_file)
inner_join_data = pd.read_csv(inner_join_file)
unique_data1 = pd.read_csv(unique_file1)
unique_data2 = pd.read_csv(unique_file2)

for data in [outer_join_data, inner_join_data, unique_data2]:
    data["SFR_mean"] = data[["SFRFUV", "SFRHa"]].mean(axis=1)
    # log SFR_mean
    data["logSFR_mean"] = np.log10(data["SFR_mean"])
    # rename columns a26_1 and a26_2 to a26_1 if they exist
    if "a26_1" in data.columns:
        data.rename(columns={"a26_1": "a26_1"}, inplace=True)
    if "a26_2" in data.columns:
        data.rename(columns={"a26_2": "a26_1"}, inplace=True)
    print(data.columns)

# Read the combinations CSV
with open(combinations_file, "r", newline="") as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        x = row["x"].strip()
        y = row["y"].strip()
        description = row["description"].strip()
        log_x = row["log_x"].strip().lower() == "true"
        log_y = row["log_y"].strip().lower() == "true"

        # Labels for the plots
        xlabel = "HECATE"
        ylabel = "UNGC"

        # ==========================
        # 1. All galaxies
        # ==========================
        plt.figure(figsize=(10, 6))
        plt.hist(
            outer_join_data[x].dropna(),
            bins=30,
            alpha=0.5,
            label=f"{xlabel} (All)",
            color="gray",
        )
        plt.hist(
            outer_join_data[y].dropna(),
            bins=30,
            alpha=0.5,
            label=f"{ylabel} (All)",
            color="blue",
        )
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.title(f"Histogram of {description} - All Galaxies")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/{description.replace(' ', '_')}_all_galaxies.png")
        plt.close()

        # ==========================
        # 2. Common galaxies
        # ==========================
        ####################################################################################
        # plt.figure(figsize=(10, 6))                                                      #
        # plt.hist(                                                                        #
        #     inner_join_data[x].dropna(),                                                 #
        #     bins=30,                                                                     #
        #     alpha=0.5,                                                                   #
        #     label=f"{xlabel} (Common)",                                                  #
        #     color="gray",                                                                #
        # )                                                                                #
        # plt.hist(                                                                        #
        #     inner_join_data[y].dropna(),                                                 #
        #     bins=30,                                                                     #
        #     alpha=0.5,                                                                   #
        #     label=f"{ylabel} (Common)",                                                  #
        #     color="blue",                                                                #
        # )                                                                                #
        # plt.xlabel(xlabel)                                                               #
        # plt.ylabel("Count")                                                              #
        # plt.title(f"Histogram of {description} - Common Galaxies")                       #
        # plt.legend()                                                                     #
        # plt.grid(True)                                                                   #
        # plt.savefig(f"{output_dir}/{description.replace(' ', '_')}_common_galaxies.png") #
        # plt.close()                                                                      #
        ####################################################################################

        # ==========================
        # 3. Unique galaxies
        # ==========================
        plt.figure(figsize=(10, 6))
        plt.hist(
            unique_data1[y].dropna(),
            bins=30,
            alpha=0.5,
            label=f"{xlabel} (Unique Catalog 1)",
            color="gray",
        )
        plt.hist(
            unique_data2[x].dropna(),
            bins=30,
            alpha=0.5,
            label=f"{ylabel} (Unique Catalog 2)",
            color="blue",
        )
        plt.xlabel(xlabel)
        plt.xscale("log" if log_x and log_y is True else "linear")
        plt.ylabel("Count")
        plt.title(f"Histogram of {description} - Unique Galaxies")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/{description.replace(' ', '_')}_unique_galaxies.png")
        plt.close()

print("Plots have been generated and saved in the 'plots' directory.")
