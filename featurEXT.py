import cv2
import numpy as np
from PIL import Image

imgO = cv2.imread("eng2.jpg", cv2.IMREAD_GRAYSCALE)
(thresh, img) = cv2.threshold(imgO, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imwrite('engOTSU2.png',img)
#get skeleton from binarised image
#we're following this link: http://download.springer.com/static/pdf/173/chp%253A10.1007%252F978-3-642-34500-5_69.pdf?originUrl=http%3A%2F%2Flink.springer.com%2Fchapter%2F10.1007%2F978-3-642-34500-5_69&token2=exp=1462706768~acl=%2Fstatic%2Fpdf%2F173%2Fchp%25253A10.1007%25252F978-3-642-34500-5_69.pdf%3ForiginUrl%3Dhttp%253A%252F%252Flink.springer.com%252Fchapter%252F10.1007%252F978-3-642-34500-5_69*~hmac=dcc429c2c5f083978f1f2070ddff716d1776a9ef7ce525b8eea438a5e33b7581
img = cv2.imread('engOTSU2.png',0)
size = np.size(img)
skel = np.zeros(img.shape,np.uint8)

ret,img = cv2.threshold(img,127,255,0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
done = False


print "Skeletenizing..."
while( not done):
    eroded = cv2.erode(img,element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(img,temp)
    skel = cv2.bitwise_or(skel,temp)
    img = eroded.copy()

    zeros = size - cv2.countNonZero(img)
    if zeros==size:
        done = True

cv2.imwrite('skel2.png',skel)

cv2.imshow("skel",skel)
cv2.waitKey(0)
cv2.destroyAllWindows()

