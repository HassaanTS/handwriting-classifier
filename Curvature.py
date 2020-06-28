import cv2
import numpy as np
from PIL import Image

# img = cv2.imread('engOTSU.png',0)
# im2, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
# cv2.imwrite('im2.png',im2)
#
#
# cv2.drawContours(im2, contours, -1, (255,0,255), 3)
# cv2.imwrite('curves.png',img)


im = Image.open("engOTSU2.png")
width, height = im.size

Orig_Image = np.array(im) # put image in array

foreG = float(0)#n2
backG = float(0)#n1

for i in range(height):
    for j in range(width):
        if (Orig_Image[i][j] == 0):
            foreG += 1
        elif (Orig_Image[i][j] == 255):
            backG += 1
print backG
print foreG
Curve = float(backG) - float(foreG)
denom = float(backG) + float(foreG)
Curve = float(Curve)/float(denom)


print "curvature:", Curve





