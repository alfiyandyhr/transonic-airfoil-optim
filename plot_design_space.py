import matplotlib.pyplot as plt
import numpy as np
from BSpline import BSplineFromControlPoints


uiuc_data = np.genfromtxt('Designs/baseline/rae2282_base.dat',
						   skip_header=1, skip_footer=66, usecols=(0,1))
uiuc_data2 = np.genfromtxt('Designs/baseline/rae2282_base.dat',
						   skip_header=67, skip_footer=0, usecols=(0,1))
uiuc_data = np.concatenate((uiuc_data, uiuc_data2), axis=0)

d_lower = np.array([0.0,0.01,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02])
d_upper = np.array([0.0,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.01,0.0])

control_base = np.genfromtxt('Designs/baseline/rae2282_base_control.dat',
							  usecols=(0,1))

control_max = np.copy(control_base)
control_min = np.copy(control_base)

control_max[0:10,1] = control_max[0:10:,1] - d_lower
control_max[10:21,1] = control_max[10:21,1] + d_upper
control_min[0:10,1] = control_min[0:10:,1] + d_lower
control_min[10:21,1] = control_min[10:21,1] - d_upper

zeros = np.zeros((len(control_base),1))

control_max = np.concatenate((control_max,zeros),axis=1)
control_min = np.concatenate((control_min,zeros),axis=1)

np.savetxt('Designs/baseline/control_max.dat',control_max)
np.savetxt('Designs/baseline/control_min.dat',control_min)

bspline = BSplineFromControlPoints(degree=3)

bspline.create(ctrlpts_path='Designs/baseline/rae2282_base_control.dat')
bspline.rotate_and_dilate(output_path='Designs/baseline/bspline_points_base.dat')
# bspline.to_pointwise_format(file_path='Designs/baseline/bspline_points_base.dat')

bspline.create(ctrlpts_path='Designs/baseline/control_max.dat')
bspline.rotate_and_dilate(output_path='Designs/baseline/bspline_points_max.dat')
# bspline.to_pointwise_format(file_path='Designs/baseline/bspline_points_max.dat')

bspline.create(ctrlpts_path='Designs/baseline/control_min.dat')
bspline.rotate_and_dilate(output_path='Designs/baseline/bspline_points_min.dat')
# bspline.to_pointwise_format(file_path='Designs/baseline/bspline_points_min.dat')

bspline_points_base = np.genfromtxt('Designs/baseline/bspline_points_base.dat')
bspline_points_max = np.genfromtxt('Designs/baseline/bspline_points_max.dat')
bspline_points_min = np.genfromtxt('Designs/baseline/bspline_points_min.dat')


control_rand = np.genfromtxt('Designs/gen_1/control_points/random_design81.dat')

bspline_points_rand = np.genfromtxt('Designs/gen_1/bspline_points/bspline_points_random81.dat',
									 skip_header=1, skip_footer=101)
bspline_points_rand1 = np.genfromtxt('Designs/gen_1/bspline_points/bspline_points_random81.dat',
									  skip_header=103, skip_footer=0)
bspline_points_rand = np.concatenate((bspline_points_rand,bspline_points_rand1),axis=0)

# ------------------------------------------------------------------------------------------------ #

plt.plot(uiuc_data[:,0], uiuc_data[:,1], 'b-', markersize = 3, label='RAE2282 - Baseline')

# plt.errorbar(control_base[0:10:,0], control_base[0:10:,1],yerr=d_lower,fmt='ro',markersize=3,ecolor='black',capsize=3, label='Control points')
# plt.errorbar(control_base[10:21:,0], control_base[10:21:,1],yerr=d_upper,fmt='ro',markersize=3,ecolor='black',capsize=3)

# plt.plot(control_base[:,0], control_base[:,1], 'bo', markersize = 2, label='Base Control Points')

# plt.plot(bspline_points_base[:,0],bspline_points_base[:,1], 'r--', label='Bspline base')
# plt.plot(bspline_points_max[:,0],bspline_points_max[:,1], 'r--', label='Max')
# plt.plot(bspline_points_min[:,0],bspline_points_min[:,1], 'g--', label='Min')

# plt.plot(bspline_points_rand[:,0],bspline_points_rand[:,1],'ro',markersize = 3, label='B Spline')
# plt.plot(control_rand[:,0],control_rand[:,1], 'ro', markersize = 3, label='Random Control Points')

plt.xlim([-0.2, 1.2])
# plt.xlim([0.8, 1.01])
# plt.ylim([-0.04, 0.06])
plt.ylim([-0.2, 0.2])
plt.title('Design Space')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="upper right")
plt.show()