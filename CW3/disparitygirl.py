import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
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

def set_numDisparities(value):
    global numDisparities
    numDisparities = max(16, value * 16)
    print(numDisparities)

def set_blockSize(value):
    global blockSize
    blockSize = max(5, value * 2 + 1)
    print(blockSize)

# depth = 1/(disparity + k)
def getDepth(disparity,k):
    depth = 1/(disparity + k)
    return depth

def set_k(value):
    global k
    k = max(0.1, value * 0.1)
    # print(k)


if __name__ == '__main__':

    # Load left image
    filename = 'girlL.png'
    imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # edge detected images
    # imgL = cv2.Canny(imgL, 100, 140)
    # cv2.imshow('imgL', imgL)
    # cv2.waitKey(0)
    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()


    # Load right image
    filename = 'girlR.png'
    imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # edge detected images
    # imgR = cv2.Canny(imgR, 100, 140)
    # cv2.imshow('imgR', imgR)
    # cv2.waitKey(0)
    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()


    # Create a window to display the image in
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)

    # Add trackbars
    cv2.createTrackbar('NumDisparities', 'Disparity', 1, 30, set_numDisparities)
    cv2.createTrackbar('BlockSize', 'Disparity', 1, 50, set_blockSize)

    # Initialize values
    numDisparities = 16
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

    cv2.destroyAllWindows()

    cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('k', 'Depth', 1, 20, set_k)
    k = 1
    while True:
        depth = getDepth(disparityImg,k)
        depth_normalized = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow('Depth', depth_normalized)
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break
    cv2.destroyAllWindows()

    # 3D plot
    imgL_color = cv2.imread('girlL.png', cv2.IMREAD_COLOR)
    imgR_color = cv2.imread('girlR.png', cv2.IMREAD_COLOR)
    imgL_blurred = cv2.GaussianBlur(imgL_color, (31, 31), 0)
    _, threshold_depth = cv2.threshold(depth_normalized, 180, 255, cv2.THRESH_BINARY)
    depth_mask = cv2.merge([threshold_depth, threshold_depth, threshold_depth])  # Make it a 3-channel mask

    foreground = cv2.bitwise_and(imgL_color, cv2.bitwise_not(depth_mask))
    cv2.imshow('foreground', foreground)
    cv2.waitKey(0)
    background = cv2.bitwise_and(imgL_blurred, depth_mask)
    cv2.imshow('background', background)
    cv2.waitKey(0)
    selective_focus = cv2.add(foreground, background)

    cv2.imshow('selective_focus', selective_focus)
    cv2.waitKey(0)
