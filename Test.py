import cv2
# Credit: https://stackoverflow.com/questions/61643039/improving-canny-edge-detection

# read image
img = cv2.imread("astro.jpg")

# convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold
thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]

# morphology edgeout = dilated_mask - mask
# morphology dilate
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

# get absolute difference between dilate and thresh
diff = cv2.absdiff(dilate, thresh)

# invert
edges = 255 - diff

# write result to disk
cv2.imwrite("cartoon_thresh.jpg", thresh)
cv2.imwrite("cartoon_dilate.jpg", dilate)
cv2.imwrite("cartoon_diff.jpg", diff)
cv2.imwrite("cartoon_edges.jpg", edges)