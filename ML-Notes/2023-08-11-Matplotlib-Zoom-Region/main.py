"""
This script generates a plot of stock prices over time, with a zoomed-in inset of a specific region.

The main plot shows the stock prices over time, with a sudden increase in prices simulated for a specific time period.
The inset plot shows a zoomed-in view of the sudden increase in prices.

The script uses the matplotlib library to generate the plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Generate example data for time and stock prices
time = np.linspace(0, 100, 200)
stock_prices = 100 + 0.5 * time + np.random.normal(0, 10, 200)

# Simulate a sudden increase in stock prices
sudden_increase_indices = np.where((time > 40) & (time < 60))
stock_prices[sudden_increase_indices] += 30

# Create the main plot
fig, ax = plt.subplots()
ax.plot(time, stock_prices, label="Stock Prices")
ax.set_xlabel("Time")
ax.set_ylabel("Stock Price")
ax.legend()

# Create an inset axes for the zoomed-in region
axins = inset_axes(ax, width="50%", height="50%", loc="lower right")
axins.plot(time, stock_prices, label="Zoomed Inset")
axins.set_xlim(40, 60)
axins.set_ylim(140, 220)
axins.set_xticklabels("")
axins.set_yticklabels("")

# Mark the zoom region in the main plot
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.tight_layout()
plt.show()
