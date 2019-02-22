

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

# generate some sample data
import scipy.misc
lena = cv2.imread("d1.png", -1).astype(float)
plt.imshow(lena)
lena
lena[lena > 1200] = 1200
lena = ((lena*1)/1200).astype("float")
lena = 1-lena


# downscaling has a "smoothing" effect
lena = scipy.misc.imresize(lena, 0.15, interp='cubic')
print lena.shape

plt.imshow(lena)
# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:lena.shape[0], 0:lena.shape[1]]
y1, zz = np.mgrid[0:127, 0:255]
print zz.shape
aa = np.ones_like(y1)*3
# create the figure
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, lena, rstride=1, cstride=1, cmap=plt.cm.gray,
                linewidth=0)
ax.plot_surface(aa, y1, zz, rstride=1, cstride=1, color="red",
                linewidth=0)

# show it
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# point = np.array([1, 2, 3])
# normal = np.array([1, 1, 2])

# # a plane is a*x+b*y+c*z+d=0
# # [a,b,c] is the normal. Thus, we have to calculate
# # d and we're set
# d = -point.dot(normal)
# print (d, point, normal)


# # create x,y
# xx, yy = np.meshgrid(range(6), range(10))
# print xx, yy
# print xx.shape, yy.shape


# # calculate corresponding z
# z = (-normal[0] * xx - normal[1] * yy - d) * 2. / normal[2]
# print z
# z[:, :] = 10
# z[0, 3] = 9
# print
# print z
# z2 = np.ones_like(z)
# # plot the surface
# plt3d = plt.figure().gca(projection='3d')
# plt3d.plot_surface(xx, yy, z)
# plt3d.plot_surface(xx, yy, z2)
# plt.show()
