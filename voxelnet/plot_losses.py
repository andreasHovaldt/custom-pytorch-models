import matplotlib.pyplot as plt 
import numpy as np 

losses = np.loadtxt("losses.txt")
plt.plot(losses)
plt.show()
