#Yasmin Esqueda Práctica 5

import cv2
import numpy as np

imagen = cv2.imread('bosqueoscuro.jpg')
retval, threshold = cv2.threshold(imagen, 12, 255, cv2.THRESH_BINARY) #arriba de 12 es blanco, y por debajo será negro

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
