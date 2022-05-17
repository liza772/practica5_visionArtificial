#Yasmin Esqueda Práctica 5

import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
from matplotlib import pylab 

fila = 2
columna = 4

img = cv2.imread('bosqueoscuro.jpg', 1)
fig = plt.figure(figsize=(10,8), constrained_layout=True)
fig.add_subplot(fila,columna,1)
plt.imshow(img)
plt.axis('off')
plt.title("Original")

fig.add_subplot(fila,columna,2)
img = cv2.imread('bosqueoscuro.jpg', 1)
ret,img = cv2.threshold(img, 125, 255 , cv2.THRESH_BINARY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.title("Threshold_binary")

fig.add_subplot(fila,columna,3)
img = cv2.imread('bosqueoscuro.jpg', 1)
ret,img = cv2.threshold(img, 125, 255 , cv2.THRESH_BINARY_INV)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.title("Binary_inv")

fig.add_subplot(fila,columna,4)
img = cv2.imread('bosqueoscuro.jpg', 1)
ret,img = cv2.threshold(img, 125, 255 , cv2.THRESH_TRUNC)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.title("Trunc")

fig.add_subplot(fila,columna,5)
img = cv2.imread('bosqueoscuro.jpg', 1)
ret,img = cv2.threshold(img, 125, 255 , cv2.THRESH_TOZERO)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.title("To Zero")

fig.add_subplot(fila,columna,6)
img = cv2.imread('bosqueoscuro.jpg', 1)
ret,img = cv2.threshold(img, 125, 255 , cv2.THRESH_TOZERO_INV)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.title("Tz_inv")

fig.add_subplot(fila,columna,7)
grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.imread('bosqueoscuro.jpg', 1)
retval2, otsu = cv2.threshold(grayscaled, 125, 255 , (cv2.THRESH_BINARY+cv2.THRESH_OTSU))
img = cv2.cvtColor(otsu, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.title("Otsu")

fig.add_subplot(fila,columna,8)
grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.imread('bosqueoscuro.jpg', 1)
etval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)
gaus = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
img = cv2.cvtColor(gaus, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.title("Gaus")


imagen = cv2.imread('bosqueoscuro.jpg')
retval, threshold = cv2.threshold(imagen, 12, 255, cv2.THRESH_BINARY) #arriba de 12 es blanco, y por debajo será negro

""""
grayscaled = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
retval2, threshold2 = cv2.threshold(grayscaled, 12, 255, cv2.THRESH_BINARY)
gaus = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
retval2, otsu = cv2.threshold(grayscaled, 125, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow('original',imagen)
cv2.imshow('threshold', threshold)
cv2.imshow('threshold2', threshold)
cv2.imshow('gaus', gaus)
cv2.imshow('otsu', otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

plt.show()