import numpy as np
import matplotlib.pyplot as plt

def delayed_tau_sfr(t_sf, A_del, tau):
    return (A_del * t_sf / tau**2) * np.exp(-t_sf / tau)

def plot_delayed_tau(A_del=1, taus=[1, 2, 3, 4, 5], t_max=10, num_points=1000):
    t_sf = np.linspace(0, t_max, num_points)  # Avoid t_sf = 0 for numerical stability
    
    plt.figure(figsize=(8, 6))
    for tau in taus:
        sfr = delayed_tau_sfr(t_sf, A_del, tau)
        plt.plot(t_sf, sfr, label=r"$\tau$ ="+f"{tau} Gyr")
        plt.axvline(tau, linestyle='--', color='gray', alpha=0.7)
        plt.scatter([tau], [delayed_tau_sfr(tau, A_del, tau)], color='grey', zorder=2)  # Add point at SFR(tau)
    
    plt.xlabel(r"$t_{sf}$ [Gyr]")
    plt.ylabel(r"$\text{SFR}_{del}(t_{sf})\ [M_\odot \cdot yr^{-1}]$")
    plt.title(r"Delayed-$\tau$ Model, for $A_{del} = 1\ M_\odot$")
    plt.legend()
    plt.grid()
    plt.savefig("images/delayed_tau_sfr.png")
    plt.close()

# Example usage
plot_delayed_tau(A_del=1, taus=[1, 2, 3, 4, 5], t_max=10)

