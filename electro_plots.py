import numpy as np
import matplotlib.pyplot as plt

# Load data from file
data = np.loadtxt("electro-validation.data")
A1, B1, A2, B2 = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

# Helper function to plot regression
def plot_regression(x, y, title):
    plt.figure()

    squared_diffs = (x - y) ** 2
    max_sq_dist = np.max(squared_diffs)

    #'--k'=black dashed line, 'yo' = yellow circle marker
    plt.plot(x, y, 'yo', label='Data points')
    line, = plt.plot(x, x, color='black', label='Ideal')

    plt.xlabel("Our Implementation (energies in Hartree)")
    plt.ylabel("Reference xTB-GFN2 (energies in Hartree)")
    plt.title(title)
    plt.legend(title=f"Max Δ²: {max_sq_dist:.2e}")

    return plt

# Plot for Metric 1
plt = plot_regression(A1, B1, "Linear Regression: Electrostatic Energies")
plt.tight_layout()
plt.savefig("es_check.png", bbox_inches='tight')
plt.close()

# Plot for Metric 2
plt = plot_regression(A2, B2, "Linear Regression: Self Consistent Charges")
plt.tight_layout()
plt.savefig("scc_check.png", bbox_inches='tight')
plt.close()
