import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the data from the CSV file
data = pd.read_csv('Facebook.csv', delimiter='\t', usecols=range(3))

# Extract the MAU and assets columns
maus = data['MAU']
assets = data['Total Assets']

# Define the Metcalfe utility function
def metcalfe_utility(x, a):
    return a * x * (x - 1) / 2

# Perform the nonlinear regression using the Metcalfe utility function
popt, pcov = curve_fit(metcalfe_utility, maus, assets)

# Calculate the R-squared value for the Metcalfe utility function fit
residuals = assets - metcalfe_utility(maus, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((assets - np.mean(assets))**2)
r_squared = 1 - (ss_res / ss_tot)

# Calculate BIC for Metcalfe utility function
n = len(maus)
k = 1
residuals_metcalfe = assets - metcalfe_func(maus, *popt_metcalfe)
mse_metcalfe = np.mean(residuals_metcalfe**2)
bic_metcalfe = n * np.log(mse_metcalfe) + k * np.log(n)

# Perform the linear regression
linfit = np.polyfit(maus, assets, 1)

# Calculate the R-squared value for the linear fit
linresiduals = assets - np.polyval(linfit, maus)
linss_res = np.sum(linresiduals**2)
linr_squared = 1 - (linss_res / ss_tot)

# Calculate BIC for linear regression
residuals_linear = assets - np.polyval(popt_linear, maus)
mse_linear = np.mean(residuals_linear**2)
bic_linear = n * np.log(mse_linear) + k * np.log(n)

# Print the R-squared values
print('R-squared for Metcalfe Utility Function:', r_squared)
print('R-squared for Linear Regression:', linr_squared)
print('-----------------------------------------------')

# Print the BIC values
print(f'BIC for Metcalfe Utility Function: {bic_metcalfe:.2f}')
print(f'BIC for Linear Regression: {bic_linear:.2f}')
print('-----------------------------------------------')

# Plot the data and fits
plt.scatter(maus, assets, label='Data')
plt.plot(maus, metcalfe_utility(maus, *popt), 'r-', label='Metcalfe utility function')
plt.plot(maus, np.polyval(linfit, maus), 'g-', label='Linear regression')
plt.xlabel('MAU')
plt.ylabel('Assets')
plt.legend()
plt.show()
