import cv2
import numpy as np
from matplotlib import pyplot as plt

def order_points(pts):
    # Step 1: Find centre of object
    center = np.mean(pts)

    # Step 2: Move coordinate system to centre of object
    shifted = pts - center

    # Step #3: Find angles subtended from centroid to each corner point
    theta = np.arctan2(shifted[:, 0], shifted[:, 1])

    # Step #4: Return vertices ordered by theta
    ind = np.argsort(theta)
    return pts[ind]

def getContours(img, orig):
    biggest = np.array([])
    maxArea = 0
    imgContour = orig.copy()  # Change - make a copy of the image to return
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    index = None
    for i, cnt in enumerate(contours):  # Change - also provide index
        area = cv2.contourArea(cnt)
        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt,0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
                index = i  # Also save index to contour

    warped = None
    if index is not None: # Draw the biggest contour on the image
        cv2.drawContours(imgContour, contours, index, (255, 0, 0), 3)

        src = np.squeeze(biggest).astype(np.float32) # Source points
        height = image.shape[0]
        width = image.shape[1]
        # Destination points
        dst = np.float32([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]])

        # Order the points correctly
        biggest = order_points(src)
        dst = order_points(dst)

        # Get the perspective transform
        M = cv2.getPerspectiveTransform(src, dst)

        # Warp the image
        img_shape = image.shape[:2][::-1]
        warped = cv2.warpPerspective(orig, M, img_shape, flags=cv2.INTER_LINEAR)

    return biggest, imgContour, warped  # Change - also return drawn image

kernel = np.ones((3,3))
image = cv2.imread('test.jpg')

imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
imgCanny = cv2.Canny(imgBlur,150,200)
imgDial = cv2.dilate(imgCanny,kernel,iterations=1)
imgThres = cv2.erode(imgDial,kernel,iterations=1)
biggest, imgContour, warped = getContours(imgThres, image)  # Change

titles = ['original', 'Blur', 'Canny', 'Dilate', 'Threshold', 'Contours', 'Warped']

images = [image[...,::-1],  imgBlur, imgCanny, imgDial, imgThres, imgContour, warped[...,::-1]]  # Change

for i in range(7):
    plt.subplot(3, 4, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    
plt.show()