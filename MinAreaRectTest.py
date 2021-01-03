import numpy as np
import tifffile as tiff
import MinAreaRectangle 
import matplotlib.pyplot as plt
import cv2 
#zeros = np.zeros((100, 100), dtype=np.uint8)
#zeros[:5,:5] = 255
tissue_sample = tiff.imread('/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/images/004dd32d9cd167d9cc31c13b704498af.tiff')
print("tissue_sample_shape " + str(tissue_sample.shape))  
tissue_sample_initial = cv2.resize(tissue_sample, (8000,8000), interpolation = cv2.INTER_NEAREST)
tissue_sample=tissue_sample_initial.astype('float64')
tissue_sample_one = np.add(tissue_sample[:,:,0],tissue_sample[:,:,1])
tissue_sample = np.add(tissue_sample_one,tissue_sample[:,:,2])
print("tissue_sample")
print(tissue_sample)
indices = np.where(tissue_sample != [765])
#print("indices: " + str(indices))
#print('  ')
coordinates = list(zip(indices[0], indices[1]))
print("number of coord non white: " + str(len(coordinates)))
print(' ')
box_stats=MinAreaRectangle.MinimumBoundingBox(coordinates)
print("end: " + str(box_stats.corner_points))
integers = []
for i in box_stats.corner_points:
	s = []
	s.append(int(i[0]))
	s.append(int(i[1]))
	s = tuple(s)
	integers.append(s)

print("new" + str(integers))

implot = plt.imshow(tissue_sample_initial)
x = []
y = []
for i in integers:
	x.append(i[0])
	y.append(i[1])

#put a blue dot at (10, 20)
plt.scatter(x=x, y=y)

plt.show()

