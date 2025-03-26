import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set working directory to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Create directories for saving plots
plot_dirs = ["gr_plots", "bu_plots", "bv_plots", "ms_plots"]
for d in plot_dirs:
    os.makedirs(d, exist_ok=True)

# Load dataset
df = pd.read_csv("joined_output_HECATE.csv")

# set ggplot style
plt.style.use('ggplot')

# Define function to create and save CMD plots
def plot_cmd(df, x_col, y_col, xlabel, ylabel, title, filename, folder, hue=None, cmap=False, invert_y=True, clabel = "Morphological Type"):
    plt.figure(figsize=(11, 6), dpi=350)
    if hue:
        if cmap:
            scatter = plt.scatter(df[x_col], df[y_col], c=df[hue], cmap="viridis", alpha=0.7, marker=".")
            plt.colorbar(scatter, label=clabel)
        else:
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, edgecolor="k", alpha=0.7, marker=".")
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col, edgecolor="k", alpha=0.7, marker=".")
    if invert_y:
        plt.gca().invert_yaxis()
    #add grid
    plt.grid(True)
    #remove title from legend
    plt.legend(title=None)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, filename), dpi = 350)
    plt.close()

# Define improved labels for CMD plots
xlabel_gr = r"$g - r$ (mag)"
ylabel_mr = r"$M_R$ (Absolute Magnitude in r-band)"
xlabel_bu = r"$B - U$ (mag)"
ylabel_mb = r"$M_B$ (Absolute Magnitude in B-band)"
xlabel_bv = r"$B - V$ (mag)"
ylabel_mb = r"$M_B$ (Absolute Magnitude in B-band)"
xlabel_ms = r"$\log M_{*}/M_\odot$"
ylabel_ms = r"$\log (\mathrm{SFR}_{\mathrm{total}} /(M_\odot$ yr$^{-1}$))"

# Categorize tau_up and tau_np for color coding
df["tau_up_category"] = pd.cut(df["tau_up"], bins=[-float("inf"), 5, 10, float("inf")], labels=[r"$\tau < 5$ Gyr", r"$5 \leq \tau \leq 10$ Gyr", r"$\tau > 10$ Gyr"])
df["tau_np_category"] = pd.cut(df["tau_np"], bins=[-float("inf"), 5, 10, float("inf")], labels=[r"$\tau < 5$ Gyr", r"$5 \leq \tau \leq 10$ Gyr", r"$\tau > 10$ Gyr"])

# Generate and save CMD and main sequence plots
plot_cmd(df, "gr", "MR", xlabel_gr, ylabel_mr, "CMD Plot (All Galaxies, g - r)", "cmd_all_gr.png", "gr_plots")
plot_cmd(df, "gr", "MR", xlabel_gr, ylabel_mr, "CMD Plot Colored by $\\tau$ (Uniform Prior, $\\tau$ in Gyr)", "cmd_tau_up_gr.png", "gr_plots", "tau_up_category")
plot_cmd(df, "gr", "MR", xlabel_gr, ylabel_mr, "CMD Plot Colored by $\\tau$ (Normal Prior, $\\tau$ in Gyr)", "cmd_tau_np_gr.png", "gr_plots", "tau_np_category")
plot_cmd(df, "gr", "MR", xlabel_gr, ylabel_mr, "CMD Plot Colored by Morphological Type", "cmd_morph_gr.png", "gr_plots", "merge_T", cmap=True)

plot_cmd(df, "bu", "MB", xlabel_bu, ylabel_mb, "CMD Plot (All Galaxies, B - U)", "cmd_all_bu.png", "bu_plots")
plot_cmd(df, "bu", "MB", xlabel_bu, ylabel_mb, "CMD Plot Colored by $\\tau$ (Uniform Prior, $\\tau$ in Gyr)", "cmd_tau_up_bu.png", "bu_plots", "tau_up_category")
plot_cmd(df, "bu", "MB", xlabel_bu, ylabel_mb, "CMD Plot Colored by $\\tau$ (Normal Prior, $\\tau$ in Gyr)", "cmd_tau_np_bu.png", "bu_plots", "tau_np_category")
plot_cmd(df, "bu", "MB", xlabel_bu, ylabel_mb, "CMD Plot Colored by Morphological Type", "cmd_morph_bu.png", "bu_plots", "merge_T", cmap=True)

plot_cmd(df, "bv", "MB", xlabel_bv, ylabel_mb, "CMD Plot (All Galaxies, B - V)", "cmd_all_bv.png", "bv_plots")
plot_cmd(df, "bv", "MB", xlabel_bv, ylabel_mb, "CMD Plot Colored by $\\tau$ (Uniform Prior, $\\tau$ in Gyr)", "cmd_tau_up_bv.png", "bv_plots", "tau_up_category")
plot_cmd(df, "bv", "MB", xlabel_bv, ylabel_mb, "CMD Plot Colored by $\\tau$ (Normal Prior, $\\tau$ in Gyr)", "cmd_tau_np_bv.png", "bv_plots", "tau_np_category")
plot_cmd(df, "bv", "MB", xlabel_bv, ylabel_mb, "CMD Plot Colored by Morphological Type", "cmd_morph_bv.png", "bv_plots", "merge_T", cmap=True)

plot_cmd(df, "logM_total", "logSFR_total", xlabel_ms, ylabel_ms, "Main Sequence Plot", "ms_plot.png", "ms_plots", invert_y=False)
plot_cmd(df, "logM_total", "logSFR_total", xlabel_ms, ylabel_ms, "Main Sequence Colored by $\\tau$ (Uniform Prior, $\\tau$ in Gyr)", "ms_tau_up.png", "ms_plots", "tau_up_category", invert_y=False)
plot_cmd(df, "logM_total", "logSFR_total", xlabel_ms, ylabel_ms, "Main Sequence Colored by $\\tau$ (Normal Prior, $\\tau$ in Gyr)", "ms_tau_np.png", "ms_plots", "tau_np_category", invert_y=False)
plot_cmd(df, "logM_total", "logSFR_total", xlabel_ms, ylabel_ms, "Main Sequence Colored by Morphological Type", "ms_morph.png", "ms_plots", "merge_T", cmap=True, invert_y=False)
plot_cmd(df, "logM_total", "logSFR_total", xlabel_ms, ylabel_ms, "Main Sequence Colored by Specific SFR", "ms_sSFR.png", "ms_plots", "logsSFR", cmap=True, invert_y=False, clabel=r"log(sSFR $\cdot$ yr)")
plot_cmd(df, "logM_total", "logSFR_total", xlabel_ms, ylabel_ms, "Main Sequence Colored by $t_{sf}$ (Normal Prior, $t_{sf}$ in Gyr)", "ms_tsf_np.png", "ms_plots", "t_sf_np", cmap=True, invert_y=False,clabel=r"$t_{sf}$ (Gyr)")
plot_cmd(df, "logM_total", "logSFR_total", xlabel_ms, ylabel_ms, "Main Sequence Colored by $t_{sf}$ (Uniform Prior, $t_{sf}$ in Gyr)", "ms_tsf_up.png", "ms_plots", "t_sf_up", cmap=True, invert_y=False,clabel=r"$t_{sf}$ (Gyr)")

print("Plots saved in respective folders.")
