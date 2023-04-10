# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 17:26:50 2023

@author: Rutho
"""

# OK Ruth is going to attempt to write her own script (but mostly copy Mathew's)

# Let's get all the packages we need first off:
    
from skimage.io import imread
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters,measure
from skimage.filters import threshold_local
from scipy.ndimage import rotate


# Get the path to your folder for saving the extra images generated, probably the path to a well:
folder_path = "C:/Users/Rutho/OneDrive/Documents/SCHOOL/Year 3/Project/Raw_Data/24-03-23_PEGReplicates/Slide1/PEG_1in2000/well-1/"

# Then you need to tell Spyder where the ingredients are in the pantry:
    # But first we define the very first variables:
        
green_image = "Left "
red_image = "Right "

    # Give it a list of paths:

path_list = []

for i in range(1, 10):
    green_path = os.path.join(folder_path, green_image + f"({i})" + ".tif")
    path_list.append(green_path)
    
    red_path = os.path.join(folder_path, red_image + f"({i})" + ".tif")
    path_list.append(red_path)
      
print(path_list)

# Next steps, we'll be choosing our ingredients and checking we've got everything:

def load_image(toload):
    
    image = imread(toload)
    
    return image

# And pressing all 50 frames of the image into a 2D one that can be further analysed:
def z_project(image):
    
    mean_int = np.mean(image,axis=0)
  
    return mean_int

# Let's do some thresholding:
    # Calculate how much of it you have first and then subtract it from the image:
        
def subtract_bg(image):
    background = threshold_local(image, 11, offset=np.percentile(image, 1), method='median')
    bg_corrected = image - background
    return bg_corrected
    
    # Calculate a threshold value for everything else based off the mean and 3*SD of the input image, and then apply this to the threshold image:

def threshold_image_std(input_image):  
    threshold_value = input_image.mean() + 3*input_image.std()
    print(threshold_value)
    binary_image = input_image > threshold_value
    return threshold_value, binary_image

# Now that you have your threshold value, you apply it to the input image:
        
def threshold_image_standard(input_image,thresh):
    binary_image = input_image > thresh
    return binary_image

# Take the input image, along with your threshold value (now defined as threshold number), and binarise the image using Otsu method

def threshold_image_fixed(input_image,threshold_number):
    threshold_value = threshold_number   
    binary_image = input_image>threshold_value

    return threshold_value, binary_image

# The next part is to label each '1' in the binary image with another unique label, so that 0*label = 0, whilst 1*label = label. This gives you another image called the 'labelled image'.

def label_image(input_image):
    labelled_image = measure.label(input_image)
    number_of_features = labelled_image.max()
 
    return number_of_features, labelled_image

# Let's say you want to take a look at these images you've been generating so far which are the binary and the labelled image:
    
    def show(input_image,color=''):
        if(color=='Red'):
            plt.imshow(input_image,cmap="Reds")
            plt.show()
        elif(color=='Blue'):
            plt.imshow(input_image,cmap="Blues")
            plt.show()
        elif(color=='Green'):
            plt.imshow(input_image,cmap="Greens")
            plt.show()
        else:
            plt.imshow(input_image)
            plt.show() 
        
# Or if you wanted to measure some more parameters based off of the images you've created, the following lines can show you Area, Perimeter, Centroid, Orientation, Axes Lengths, and Intensities. 
    # (This would be done using the labelled and original images.)

def analyse_labelled_image(labelled_image,original_image):
    measure_image = measure.regionprops_table(labelled_image,intensity_image=original_image, properties = ('area', 'perimeter', 'centroid', 'orientation', 'major_axis_length', 'minor_axis_length', 'mean_intensity', 'max_intensity'))
    measure_dataframe = pd.DataFrame.from_dict(measure_image)
    return measure_dataframe

# Now we move onto coincidence analysis!
    # Start by looking at coincidence in terms of pixels only between the binary images you generated:
        # The calculations in short: '&' performs a bit-by-bit operation, counting how many pixels overlap between binary images 1 and 2 (from green and red, respectively) to make a new binary image, and then counts the number of pixels with value 1. 
        
Output_all = pd.DataFrame(columns=['Number green', 'Number red', 'Number coincident', 'Number chance', 'Q'])

for path in path_list:
    def coincidence_analysis_pixels(binary_image1,binary_image2):
        pixel_overlap_image = binary_image1 & binary_image2         
        pixel_overlap_count = pixel_overlap_image.sum()
        pixel_fraction = pixel_overlap_image.sum()/binary_image1.sum()
    
        return pixel_overlap_image, pixel_overlap_count, pixel_fraction

    # Next we check coincidence in terms of features, using this new binary image and the labelled image:
        
def feature_coincidence(binary_image1,binary_image2):
    number_of_features, labelled_image1 = label_image(binary_image1)       
    coincident_image = binary_image1 & binary_image2 
    coincident_labels = labelled_image1*coincident_image
    coinc_list, coinc_pixels = np.unique(coincident_labels, return_counts = True)
    total_labels = labelled_image1.max()
    total_labels_coinc = len(coinc_list)
    fraction_coinc = total_labels_coinc/total_labels
    
    label_list, label_pixels = np.unique(labelled_image1,       return_counts = True)
    fract_pixels_overlap = []
    for i in range(len(coinc_list)):
        overlap_pixels = coinc_pixels[i]
        label = coinc_list[i]
        total_pixels = label_pixels[label]
        fract = 1.0*overlap_pixels/total_pixels
        fract_pixels_overlap.append(fract)
        
        #  Label of the background features (label 0) is replaced with a very large value of 1 000 000 to exclude it from the 'np.isin' operation of the next line. Once you've run that, you can replace it back to 0 again. You now have an image that shows just the coincident features i.e. a synaptosome.
        
    coinc_list[0] = 1000000 
    coincident_features_image = np.isin(labelled_image1, coinc_list) 
    coinc_list[0] = 0
    non_coincident_features_image = ~np.isin(labelled_image1, coinc_list)
        
    
    
    return coinc_list, coinc_pixels,fraction_coinc, coincident_features_image, non_coincident_features_image,  fract_pixels_overlap

# From here we now look at chance coincidence. We do this by rotating one of the images to see if any features overlap anyway. 
    # First step is rotate! Create a temporary matrix, appending each element of the original image to it (twice!), which should give an image rotated 180 degrees clockwise.

def rotate(matrix):
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    size = max(n_rows, n_cols)
    square_matrix = [[0]*size for _ in range(size)]
    for i in range(n_rows):
        for j in range(n_cols):
            square_matrix[i][j] = matrix[i][j]
    temp_matrix = []
    column = size - 1
    for column in range(size):
        temp = []
        for row in range(size-1, -1, -1):
            temp.append(square_matrix[row][column])
        temp_matrix.append(temp)
    for i in range(n_rows):
        for j in range(n_cols):
            matrix[i][j] = temp_matrix[i][j]
    return matrix


for i in range(0, len(path_list), 2):
    left_path = path_list[i]
    right_path = path_list[i+1]
    
    left_image = load_image(left_path)
    right_image = load_image(right_path)
    
    green_flat = np.mean(left_image, axis=0)
    red_flat = np.mean(right_image, axis=0)
    
    green_bg_remove = subtract_bg(green_flat)
    red_bg_remove = subtract_bg(red_flat)
    
    thr_left, green_binary = threshold_image_std(green_bg_remove)
    thr_right, red_binary = threshold_image_std(red_bg_remove)

    # Use green for left image and red for right image
    green = green_binary
    red = red_binary
    
    
# Save all that hard work:
    
    imsr = Image.fromarray(green_bg_remove)
    imsr.save(path_list[i] + '_BG_Removed.tif')
    
    imsr = Image.fromarray(red_bg_remove)
    imsr.save(path_list[i+1] + '_BG_Removed.tif')
    
    
    imsr = Image.fromarray(green_binary)
    imsr.save(path_list[i] + '_Binary.tif')
    
    imsr = Image.fromarray(red_binary)
    imsr.save(path_list[i+1] + '_Binary.tif')
    
# Do the coincidence analysis:
    # For the green:
        
    number_green, labelled_green = label_image(green_binary)
    print("%d features were detected in the green image." % number_green)
    measurements_green = analyse_labelled_image(labelled_green, green_flat)
    
    # And for the red:
        
    number_red, labelled_red = label_image(red_binary)
    print("%d features were detected in the red image." % number_red)
    measurements_red = analyse_labelled_image(labelled_red, red_flat)

    # Altogether now:
        
green_coinc_list, green_coinc_pixels, green_fraction_coinc, green_coincident_features_image, green_non_coincident_features_image, green_fract_pixels_overlap = feature_coincidence(green_binary, red_binary)

red_coinc_list, red_coinc_pixels,red_fraction_coinc, red_coincident_features_image, red_non_coincident_features_image, red_fract_pixels_overlap = feature_coincidence(red_binary, green_binary)

number_of_coinc = len(green_coinc_list)

# Now what if you have a high coincidence value due to high density?

green_binary_rot = rotate(green_binary) 

chance_coinc_list, chance_coinc_pixels, chance_fraction_coinc, chance_coincident_features_image, chance_non_coincident_features_image, chance_fract_pixels_overlap=feature_coincidence(green_binary_rot, red_binary)

number_of_chance=len(chance_coinc_list)

# The association quotient (Q) is calculated by: Q = (Co-Ch)/[(G+R)-(Co-Ch)] where:
    # Co = number of coincident points
    # Ch = number of chance coincident points
    # G = number of green features
    # R = number of red features

Q = (number_of_coinc - number_of_chance) / (number_green + number_red - (number_of_coinc - number_of_chance))

    # Save the last image which contains just the coincident features between the red and green images:
        
imsr = Image.fromarray(green_coincident_features_image)
imsr.save(path_list[i] + '_Coincident.tif')

imsr = Image.fromarray(red_coincident_features_image)
imsr.save(path_list[i+1] + '_Coincident.tif')

# Output everything and save a .csv file of your analysis:
    
Output_all = Output_all.append({'Number green' : number_green, 
                                    'Number red' : number_red, 
                                    'Number coincident' : number_of_coinc, 
                                    'Number chance' : number_of_chance, 
                                    'Q' : Q}, ignore_index=True)



Output_all.to_csv(folder_path + 'All.csv', sep = '\t')

for path in path_list:
    number_green = ...
    number_red = ...
    number_of_coinc = ...
    number_of_chance = ...
    Q = ...
    
    print('Path:', path)
    print('Number green:', number_green)
    print('Number red:', number_red)
    print('Number coincident:', number_of_coinc)
    print('Number chance:', number_of_chance)
    print('Q:', Q)
    
    Output_all = Output_all.append({'Number green': number_green,
                                    'Number red': number_red,
                                    'Number coincident': number_of_coinc,
                                    'Number chance': number_of_chance,
                                    'Q': Q}, ignore_index=True)
