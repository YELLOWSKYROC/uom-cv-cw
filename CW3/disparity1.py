import numpy as np
import cv2
import sys
# from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

# ================================================
#
def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1 # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0 # Map is fixed point int with 4 fractional bits

    return disparity # floating point image
# ================================================
 
#  trackbars
def set_numDisparities(value):
    global numDisparities
    numDisparities = max(16, value * 16)
    print(numDisparities)

def set_blockSize(value):
    global blockSize
    blockSize = max(5, value * 2 + 1)
    print(blockSize)

# ================================================
#
# def plot(disparity, baseline, f, doffs):
#     # This just plots some sample points. Change this function to
#     # plot the 3D reconstruction from the disparity map and other values
#     h, w = disparity.shape
#     x = np.zeros((h, w))
#     y = np.zeros((h, w))
#     z = np.zeros((h, w))

#     for i in range(h):
#         for j in range(w):
#             d = disparity[i, j]
#             temp = baseline * (f / (d + doffs))
#             if 5700 < temp < 8000:
#                 z[i, j] = baseline * (f / (d + doffs))
#                 x[i, j] = (j * z[i, j]) / f
#                 y[i, j] = (i * z[i, j]) / f

#     # Plot 3D reconstruction
#     ax = plt.axes(projection ='3d')

#     # first view
#     ax.view_init(elev=25, azim=-15, vertical_axis='y')
#     # # second view
#     # ax.view_init(elev=180, azim=-90)
#     # # third view
#     # ax.view_init(elev=180, azim=90, vertical_axis='y')

#     # Customize the Z-axis range and intervals
#     z_min, z_max, z_interval = 5500, 8000, 250
#     ax.set_zlim(z_max, z_min)
#     ax.set_zticks(np.arange(z_max + z_interval, z_min, -z_interval))
#     ax.scatter(x, y, z, 'green', s = 0.5)

#     # Labels
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.show()

# ================================================
#
if __name__ == '__main__':

    # Load left image
    filename = 'umbrellaL.png'
    imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    imgL = cv2.Canny(imgL, 100, 140)
    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    # Load right image
    filename = 'umbrellaR.png'
    imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.Canny(imgR, 100, 140)
    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    # Create a window to display the image in
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)

    # Add two trackbars
    cv2.createTrackbar('NumDisparities', 'Disparity', 4, 30, set_numDisparities)
    cv2.createTrackbar('BlockSize', 'Disparity', 2, 50, set_blockSize)

    # # Initialize values
    numDisparities = 64
    blockSize = 5

    while True:
        # Get disparity map
        disparity = getDisparityMap(imgL, imgR, numDisparities, blockSize)

        # Normalise for display
        disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))

        # Show result
        cv2.imshow('Disparity', disparityImg)

        # Wait for spacebar press or escape before closing,
        # otherwise window will close without you seeing it
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break

    # # Show 3D plot of the scene
    # baseline = 174.019 # mm
    # f = 5806.559 # pixels
    # doffs = 114.291 # pixels

    # plot(disparity, baseline, f, doffs)

    cv2.destroyAllWindows()