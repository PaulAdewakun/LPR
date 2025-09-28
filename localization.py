from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

# Brings in the car.jpg file into the program and converts it to gray scale, which is just a 2D array
car_image = imread("car.jpg", as_gray = True)

#prints out a 2D array
print(car_image.shape)

gray_scale_car = car_image * 255
fig, (axis1, axis2) = plt.subplots(1,2)
axis1.imshow(gray_scale_car, cmap="gray")

# returns a binarized version of the gray_scale_car image for  
threshold_value = threshold_otsu(gray_scale_car)

#creates a 2D boolean array that is compareed with the threshold_value, if the pixel in the gray_scale_car array is greater than 
# the threshold_value than True is stored in the boolean which will later be linked to the color white and vice versa for balck 
binary_car_image = gray_scale_car > threshold_value
axis2.imshow(binary_car_image, cmap="gray")
plt.show()



