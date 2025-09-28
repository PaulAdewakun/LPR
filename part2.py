from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import localization

image_grouping = measure.label(localization.binary_car_image)

plate_dimensions= (0.05*image_grouping.shape[0], 0.4*image_grouping.shape[0],0.05*image_grouping.shape[1], 0.6*image_grouping.shape[1])
min_height, max_height, min_width, max_width = plate_dimensions
plate_coordinates = []
similar_objects = []
fig, (ax1) = plt.subplots(1)
ax1.imshow(localization.gray_scale_car, cmap="gray")

for region in regionprops(image_grouping):
    #filters out insignificant pixel groupings which may be noise or things other than a license plate
    if region.area<50:
        continue
    
    minRow, minCol, maxRow, maxCol = region.bbox
    region_height = maxRow - minRow
    region_width = maxCol - minCol

    if (min_height <= region_height <= max_height and min_width  <= region_width  <= max_width and region_width > region_height):
        similar_objects.append(localization.binary_car_image[minRow:maxRow, minCol:maxCol])
        plate_coordinates.append((minRow,minCol,maxRow,maxCol))
        rectBorder = patches.Rectangle((minCol, minRow), maxCol-minCol, maxRow-minRow, edgecolor="red", linewidth=2, fill= False) 
        ax1.add_patch(rectBorder)

plt.show()