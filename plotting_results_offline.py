import matplotlib.pyplot as plt

# Results PF

# FIGURE 12
# PF Result
# Line Plot
N = [1000,2000,3000,4000,5000]
#T = [10.62,10.54,10.49,10.72,10.34,10.52,10.30]
T = [19.86511206626892, 33.76174569129944, 48.24333190917969, 63.284931898117065, 78.26801443099976]
plt.figure(figsize=(900/100, 900/100), dpi=100)  # 900x900 pixels
# plt.title('Figure 12')
plt.plot(N, T, color='blue',marker='s')
plt.xlabel('Total Number of Particles')
plt.ylabel('Execution Time PF(seconds)')
plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines (dashed, slightly transparent)
plt.savefig('Figures_for_the_paper/EXecution_Time_PF.png', dpi=400)

#plt.legend()
plt.show()


#  FIGURE 13
# PF RMSE PLOTS
# Line Plot
#y= [34.18,34.17,34.17,34.17,34.03,34.03,34.03]
y = [34.03759437,34.03477807,34.04155998,34.03450482,34.03715373]
x = [1000,2000,3000,4000,5000]
plt.figure(figsize=(900/100, 900/100), dpi=100)  # 900x900 pixels
# plt.title('Figure 13')
plt.plot(x, y, label='Line',color='red',marker='x')
#plt.title('Line Plot Example')
plt.xlabel('Number of Particles')
plt.ylabel('RMSE Latitude PF')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines (dashed, slightly transparent)
# plt.ylim([0, 1.25])
plt.savefig('Figures_for_the_paper/pf_RMSE_Latitude_GPU.png', dpi=400)
plt.show()



# FIGURE 14
# pf RMSE PLOTS

# Line Plot
#y= [43.00,42.99,42.99,42.95,42.95,42.95,42.95]
y = [42.95346827,42.95346827,42.95810933,42.9568862,42.95550351]
x = [1000,2000,3000,4000,5000]
plt.figure(figsize=(900/100, 900/100), dpi=100)  # 900x900 pixels
# plt.title('Figure 14')
plt.plot(x, y, label='Line',color='red',marker='x')
#plt.title('Line Plot Example')
plt.xlabel('Number of Particles')
plt.ylabel('RMSE Longitude PF')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines (dashed, slightly transparent)
# plt.ylim([0, 1.25])
plt.savefig('Figures_for_the_paper/pf_RMSE_Longitude_GPU.png', dpi=400)
plt.show()


## Plot FIGURE 15

# Line Plot
x = [1, 2, 3, 4,5,6,7,8,9]
y1 =[0.000142,0.000144,0.000132,0.000147,0.000155,0.000142,0.000132,0.000127,0.000142]
y2 = [0.0000543,0.000055,0.0000546,0.0000544,0.0000546,0.0000542,0.0000546,0.0000541,0.000054]
y3 = [0.0004116535186767578,0.00037508010864257812,0.0003592967987060547,0.0003504037857055664,0.0003559589385986328, 0.00035331249237060547,0.00033483505249023438,0.0003439188003540039,0.0003335237503051758]


plt.figure(figsize=(900/100, 900/100), dpi=100)  # 900x900 pixels
# plt.title('Figure 15')
plt.title("Computation time Comparison for PF prediction")
plt.plot(x, y1,label="CPU time", color='blue',marker="s")
plt.plot(x, y2,label="GPU time", color='red',marker="s")
plt.plot(x, y3,label="NVIDIA Jetson Xavier", color='green',marker="s")

plt.xlabel('Iteration Number')
plt.ylabel('Execution Time per Data Point PF')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines (dashed, slightly transparent)
#plt.ylim([0, 0.0003])
plt.savefig('Figures_for_the_paper/pf_time_cuda_GPU.png', dpi=400)
plt.show()