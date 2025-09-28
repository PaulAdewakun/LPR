import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skimage.io import imread
from skimage.filters import threshold_otsu

numbers = list(range(0,10))
characters = [str(num) for num in numbers]
characters.extend(list(string.ascii_uppercase))
characters.extend(list(string.ascii_lowercase))

def read_training_data(training_directory):
    image_data = []
    target_data =[]
    for each_letter in characters:
        for each in range(10):
            image_path = os.path.join(training_directory, each_letter, each_letter + '
            img_details = imread(image_path, as_gray = True)
            binary_image = img_details <threshold_otsu(img_details)
            flat_image = binary_image.reshape(-1)
            image_data.append(flat_image)
            target_data.append(each_letter)
    
    return (np.array(image_data), np.array(target_data))

def cross_validation(model, num_of_fold, train_data, train_label):
    accuracy_result = cross_val_score(model, train_data, train_label, cv=num_of_fold)
    print("Cross Validation Result for ", str(num_of_fold), " -fold")
    print(accuracy_result * 100)
    
current_dir = os.path.dirname(os.path.realpath(__file__))
training_dataset_dir = os.path.join(current_dir, 'train')
image_data, target_data = read_training_data(training_dataset_dir)

svc_model.fit(image_data, target_data)

save_directory = os.path.join(current_dir, 'models/svc/')
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

joblib.dump(svc_model, save_directory+'/svc.pkl')