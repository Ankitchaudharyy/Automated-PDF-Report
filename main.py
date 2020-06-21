import os
import sys
import cv2
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import tensorflow_hub as hub

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
parser.add_argument('--rootdir', required=False,
                    type=str,
                    default=os.getcwd(),
                    metavar="/path/to/root/directory/Tom-Jerry-Emotion-TF-2.1",
                    help='Directory of the Toon EmotionsFolder')
parser.add_argument('--maskrcnnweight', required=False,
                    type=str,
                    metavar="/path/to/maskrcnn/weights.h5",
                    default="./models/mask_rcnn_characters_0049.h5",
                    help="Path to weights .h5 file for Mask R-CNN")
parser.add_argument('--emotionweight', required=False,
                    type=str,
                    metavar="/path/to/emotiondetectionmodel/weights.h5",
                    default="./models/20200504-001719-final-8100-image-trained-model-VGG.h5",
                    help="Path to weights .h5 file for Emotion Detection Model")
parser.add_argument('--imagefolder', required=False,
                    type=str,
                    metavar="path to image folder of extracted frames",
                    help='Images to apply the emotion detection on')
parser.add_argument('--video', required=False,
                    type=str,
                    metavar="path to video",
                    help='Video to apply the emotion detection on on')
parser.add_argument('--output', required=False,
                    type=str,
                    metavar="path to extracted images",
                    default="./extracted_test_images/",
                    help='path to save extracted images')
parser.add_argument('--savecsv', required=False,
                    type=str,
                    metavar="path to save csv",
                    help='Path to save predictions in csv format')
args = parser.parse_args()
assert args.imagefolder or args.video,\
      "Provide --image or --video to apply Emotion Detection"

# Import Mask RCNN
sys.path.append(args.rootdir)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
# from ../Mask_RCNN-2.1/utils.py import utils
from samples.characters import TomJerry

#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(args.rootdir, "model")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
CHARACTER_WEIGHTS_PATH = args.maskrcnnweight

tf.compat.v1.enable_eager_execution()

import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

config = TomJerry.CharacterConfig()
#CHARACTER_DIR = os.path.join(rootdir, 'json_data')

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# Set path to balloon weights file

# Download file from the Releases page and set its path
# https://github.com/matterport/Mask_RCNN/releases
# weights_path = "/path/to/mask_rcnn_balloon.h5"

# Or, load the last model you trained
# Load weights
print("Loading weights ", args.maskrcnnweight)
model.load_weights(args.maskrcnnweight, by_name=True)

def getopencvframes(videoFilePath):
  saveFolder = args.rootdir + '/test_folder/'

  cap = cv2.VideoCapture(videoFilePath)
  frameRate = cap.get(5)

  i=0
  count = 0
  while(cap.isOpened()):

    cap.set(cv2.CAP_PROP_POS_FRAMES, count)

    count += math.floor(frameRate)
    print(count)

    ret, frame = cap.read()
    # cv2.imshow('frame',frame)

    if (ret != True) or cv2.waitKey(1) & 0xFF == ord('q') or count>=8912:
      break

    cv2.imwrite(saveFolder+"frame"+str(i)+".jpg",frame)
    i+=1
    # if count % math.floor(frameRate) == 0:
  cap.release()
  cv2.destroyAllWindows()
if(args.video):
  videoFilePath = args.video
  getopencvframes(videoFilePath)

global tom_count
global jerry_count 
tom_count = 0
jerry_count = 0

# function to check for image aspect ratio
from PIL import Image

def make_square(im, min_size=256, fill_color=(0, 0, 0, 255)):
    '''
    function to check for image aspect ratio
    Input - im: Image in PIL Image format.
    '''
    x, y = im.size
    min_size = max(x, y)
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

# Function that returns a dictionary of class_ids as indexes and list of scores for the corresponding class_ids as values
def get_class_id_score_dict(class_ids, scores):

  '''Function that returns a dictionary of class_ids as indexes and list of 
     scores for the corresponding class_ids as values'''

  id_score_tuple_list = []
  for id, score in zip(class_ids, scores):
    id_score_tuple_list.append((id, score))
  
  from collections import defaultdict
  d1=defaultdict(list)
  for k,v in id_score_tuple_list:
    d1[k].append(v)
  d1 = dict(d1)
  return d1

# Classifer Function that returns the mask array of specificly Tom else Jerry else Unknown
def get_specific_mask(bbox, masks,  dictionary, class_ids_arr, image, image_name):
  '''
  Classifer Function that returns the mask array of specificly Tom else Jerry 
  else Unknown.

  Input: 
  masks - Array of all masks returned by detect function in model.py
  dictionary - dict of class_ids and their respective scores

  Returns: 
  Returns Array of either tom's mask if found else jerry's mask and if neither 
  of the above two is passed then returns unknown.
  '''
  path_to_save_face_black = args.output
  global tom_count
  global jerry_count

  class_ids_array = class_ids_arr

  if 1 in dictionary:
    tom_count += 1
    final_tom_index, = np.where(class_ids_array == 1)
    final_tom_index2 = final_tom_index[0]
    tom_mask =  masks[:,:,final_tom_index2]
    big_frame_tom_mask = get_mask(tom_mask, image)

    [y1, x1, y2, x2] = bbox[final_tom_index2]

    crop_image_black = big_frame_tom_mask[y1:y2, x1:x2]

    pil_crop_image_black = Image.fromarray(crop_image_black)
    new_image = make_square(pil_crop_image_black)
    new_image = np.array(new_image)
    resized_image = cv2.resize(new_image, (256, 256), 
                           interpolation=cv2.INTER_NEAREST)
    crop_image_black = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    cv2.imwrite(path_to_save_face_black +  os.path.splitext(image_name)[0] + '_tom' + '.png', crop_image_black)

  elif 2 in dictionary:
    jerry_count += 1
    final_jerry_index, = np.where(class_ids_array == 2)
    final_jerry_index2 = final_jerry_index[0]
    jerry_mask =  masks[:,:,final_jerry_index2]
    big_frame_jerry_mask = get_mask(jerry_mask, image)

    [y1, x1, y2, x2] = bbox[final_jerry_index2]

    crop_image_black = big_frame_jerry_mask[y1:y2, x1:x2]

    pil_crop_image_black = Image.fromarray(crop_image_black)
    new_image = make_square(pil_crop_image_black)
    new_image = np.array(new_image)
    resized_image = cv2.resize(new_image, (256, 256), 
                           interpolation=cv2.INTER_NEAREST)
    crop_image_black = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    cv2.imwrite(path_to_save_face_black +  os.path.splitext(image_name)[0] + '_jerry' + '.png', crop_image_black)
  else:
    img = np.zeros((256,256,3), np.uint8)
    cv2.imwrite(path_to_save_face_black + os.path.splitext(image_name)[0] + '_unknown' + '.png', img)

def get_mask(specific_mask_array, image):
  '''
  Function to return masks generated of the same size as that of original image
  Input:  specific_mask_array generated by get_specific_mask() function 
  '''
  temp = image
  temp = np.array(temp)
  for j in range(temp.shape[2]):
      temp[:,:,j] = temp[:,:,j] * specific_mask_array
  return temp

# Function to load all images and detect tom/jerry get the mask resize it and then finally save the images to generate data for emotion prediction model
def detect_crop_save(root_folder_path):

  '''
  Function to load all images and detect tom/jerry get the mask resize it and 
  then finally save the images to generate data for emotion prediction model
  '''
  for image in sorted_alphanumeric(os.listdir(root_folder_path)):
    image_path = os.path.join(root_folder_path, image)

    image2 = cv2.imread(image_path)
    image2 = np.array(image2)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image2.astype(np.float32)
    
    image7, window, scale, padding = utils.resize_image(
          image2,
          min_dim=config.IMAGE_MIN_DIM,
          max_dim=config.IMAGE_MAX_DIM,
          padding=config.IMAGE_PADDING)

    image7 = np.array(image7)
    r = model.detect([image7])[0]
    dictionary = get_class_id_score_dict(r['class_ids'], r['scores'])
    get_specific_mask(r['rois'], r['masks'], dictionary, r['class_ids'], image7, image)


    
def get_all_bbox_and_save(bbox, masks,  dictionary, class_ids_arr, image):
  '''
  Function that does everything.
  Input: bbox = r['rois'], bbox coordinates,
         masks = r['masks'] masks array,
         dictionary = dict containing info of class with their scores,
         class_ids_arr = [1 2],
         image = Original Image (1024*1024)

   Returns: Saves the Tom/Jerry cropped-resized (264*264) mask in their 
            respected Folders.   
  '''

  path_to_save_tom_face = './Frames/extracted_images/tom/'
  path_to_save_jerry_face = './Frames/extracted_images/jerry/'

  path_to_save_tom_face_black = './Frames/extracted_images/tom_black/'
  path_to_save_jerry_face_black = './Frames/extracted_images/jerry_black/'

  global tom_count
  global jerry_count

  class_ids_array = class_ids_arr

  if 1 in dictionary:
    tom_count += 1
    final_tom_index, = np.where(class_ids_array == 1)
    final_tom_index2 = final_tom_index[0]
    tom_mask =  masks[:,:,final_tom_index2]
    big_frame_tom_mask = get_mask(tom_mask, image)

    [y1, x1, y2, x2] = bbox[final_tom_index2]

    crop_img = image[y1:y2, x1:x2]

    pil_crop_image = Image.fromarray(crop_img)
    new_image = make_square(pil_crop_image)
    new_image = np.array(new_image)
    resized_image = cv2.resize(new_image, (256, 256), 
                           interpolation=cv2.INTER_NEAREST)
    crop_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    crop_image_black = big_frame_tom_mask[y1:y2, x1:x2]

    pil_crop_image_black = Image.fromarray(crop_image_black)
    new_image = make_square(pil_crop_image_black)
    new_image = np.array(new_image)
    resized_image = cv2.resize(new_image, (256, 256), 
                           interpolation=cv2.INTER_NEAREST)
    crop_image_black = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    cv2.imwrite(path_to_save_tom_face +  str(tom_count) + '.png', crop_img)
    cv2.imwrite(path_to_save_tom_face_black +  str(tom_count) + '.png', crop_image_black)
    print(f'Tom cropped face saved! at {tom_count}.png')

  if 2 in dictionary:
    jerry_count += 1
    final_jerry_index, = np.where(class_ids_array == 2)
    final_jerry_index2 = final_jerry_index[0]
    jerry_mask =  masks[:,:,final_jerry_index2]
    big_frame_jerry_mask = get_mask(jerry_mask, image)

    [y1, x1, y2, x2] = bbox[final_jerry_index2]

    crop_img = image[y1:y2, x1:x2]

    pil_crop_image = Image.fromarray(crop_img)
    new_image = make_square(pil_crop_image)
    new_image = np.array(new_image)
    resized_image = cv2.resize(new_image, (256, 256), 
                           interpolation=cv2.INTER_NEAREST)
    crop_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    crop_image_black = big_frame_jerry_mask[y1:y2, x1:x2]

    pil_crop_image_black = Image.fromarray(crop_image_black)
    new_image = make_square(pil_crop_image_black)
    new_image = np.array(new_image)
    resized_image = cv2.resize(new_image, (256, 256), 
                           interpolation=cv2.INTER_NEAREST)
    crop_image_black = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    cv2.imwrite(path_to_save_jerry_face +  str(jerry_count) + '.png', crop_img)
    cv2.imwrite(path_to_save_jerry_face_black +  str(jerry_count) + '.png', crop_image_black)
    print(f'Jerry cropped face saved! at {jerry_count}.png')
  
  print('\n')

# get the masks of both tom and jerry is available and save iy in extracted_images directory for creating dataset for emotion prediction model
def get_all_masks_and_save(masks, dictionary, class_ids_arr, image):
  '''
  get the masks of both tom and jerry is available and save iy in extracted_images
  directory for creating dataset for emotion prediction model
  '''
  path_to_save_tom_face = './Frames/extracted_images/tom/'
  path_to_save_jerry_face = './Frames/extracted_images/jerry/'

  global tom_count
  global jerry_count
  class_ids_array = class_ids_arr
  if 1 in dictionary:
    tom_count += 1
    # Found Tom Mask in the image
    # Get the max score index if more than 1 toms are detected

    # tom_max_score_index = np.array(dictionary[1]).argmax()
    # # final_tom_index = np.array(dictionary[1]).argmax()
    # if 2 in dictionary:
    #   final_tom_index = len(dictionary[2]) + tom_max_score_index
    # else:
    #   final_tom_index = tom_max_score_index  
    final_tom_index, = np.where(class_ids_array == 1)
    # # Generate mask for Tom
    final_tom_index2 = final_tom_index[0]
    tom_mask =  masks[:,:,final_tom_index2]
    # return tom_mask
    big_frame_tom_mask = get_mask(tom_mask, image)
    crop_mask(big_frame_tom_mask, path_to_save_tom_face +  str(tom_count) + '.png')

  if 2 in dictionary:
    jerry_count += 1
    # Found Jerry Mask in the image
    # Get the max score index if more than 1 Jerrys are detected
    # final_jerry_index = np.array(dictionary[2]).argmax()
    final_jerry_index, = np.where(class_ids_array == 2)
    # Generate mask for Jerry
    final_jerry_index2 = final_jerry_index[0]
    jerry_mask =  masks[:,:,final_jerry_index2]
    big_frame_jerry_mask = get_mask(jerry_mask, image)
    crop_mask(big_frame_jerry_mask, path_to_save_jerry_face + str(jerry_count) + '.png')


detect_crop_save(args.imagefolder)

all_emotions = ['Happy', 'Sad', 'Angry', 'Surprised']

IMG_SIZE = 224
def preprocessor(image_path, image_size = IMG_SIZE):
  '''
  Takes an image file path and turns it into a tensor, normalizes and reshapes it
  '''
  # Read an image file
  image = tf.io.read_file(image_path)

  # Turn the jpg image into a tensor with 3 color channels
  image = tf.image.decode_jpeg(image, channels = 3)

  # Normalization, from 0-255 to 0-1
  image = tf.image.convert_image_dtype(image, tf.float32)

  # Resize the image to our desired values (60, 60)
  image = tf.image.resize(image, size = [IMG_SIZE, IMG_SIZE])

  return image

# Create a function to return (image, label)
def get_image_label(image_path, label):
  '''
  Takes image file path and the associated label as the imput and gives out
  the tuple of the two
  '''
  image = preprocessor(image_path)
  return image, label

# Defining the batch size
BATCH_SIZE = 32
# Creating a function to convert data into batches
def create_data_batches(x, y = None, batch_size = BATCH_SIZE, valid_data = False, test_data = False):
  '''
  Creates batches of Data out of images x and labels y pairs.
  Shuffles the training data but not validation data
  Also accepts test data as input (without labels)
  '''

  if test_data:
    print('Creating Test Data Batches!')

    # Turn filepaths (images) into a list of tensors/slices a big  tensor into smaller objects
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))

    # Preprocess these sliced tensors(tensor for each image in file_names array) & batch them
    data_batch = data.map(preprocessor).batch(BATCH_SIZE)

    return data_batch

  elif valid_data:
    print('Creating Valid Data batches!')
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x),
                                              tf.constant(y)))
    data_batch = data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch

  else:

    print('Creating Training data batches!')
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x),
                                              tf.constant(y)))
    # Shuffling pathnames and labels before mapping preprocessor function
    data = data.shuffle(buffer_size = len(x))

    data_batch = data.map(get_image_label).batch(BATCH_SIZE)

  return data_batch

# Loading a saved trained model 
def load_model(model_path):
  print(f'Loading a saved model from the path: {model_path}')
  model = tf.keras.models.load_model(model_path,
                                     custom_objects = {'KerasLayer': hub.KerasLayer})
  return model

loaded_model = load_model(args.emotionweight)

test_path = args.output
test_filenames = [test_path + fname for fname in os.listdir(test_path)]
sorted_test_filenames = sorted_alphanumeric(test_filenames)

# Making Function for test Dataset predictions
def get_test_predictions(test_folder_path):

  test_path = test_folder_path
  test_filenames = [test_path + fname for fname in os.listdir(test_path)]
  sorted_test_filenames = sorted_alphanumeric(test_filenames)

  test_data = create_data_batches(x = sorted_test_filenames, test_data=True)

  test_predictions = loaded_model.predict(test_data)

  return test_data, test_predictions

# Moving the unknown extracted test images to a different folder
unknown_extracted_test_images_folder_path = args.rootdir+'/unknown_extracted_test_images' 

for file_name in sorted_alphanumeric(os.listdir(test_path)):
  if 'unknown' in file_name:
    unknown_image_source_path = os.path.join(test_path, file_name)
    unknown_image_dest_path = os.path.join(unknown_extracted_test_images_folder_path, file_name)

    os.replace(unknown_image_source_path, unknown_image_dest_path)

# Calling the get_test_predictions and prepare_dataframe functions on the segmented test data
# test_batched_data, test_segmented_predictions = get_test_predictions(test_folder_path = test_path)
test_batched_data, test_segmented_predictions = get_test_predictions(test_folder_path = test_path)

unique_labels = ['angry_jerry', 'angry_tom', 'happy_jerry', 'happy_tom',
       'sad_jerry', 'sad_tom', 'surprise_jerry', 'surprise_tom']
unique_labels = np.array(unique_labels)   


#test_segmented_predictions[:10], test_segmented_predictions.shape

# Turn prediction probabilities into their respective label (easier to understand)
def get_pred_label(prediction_probabilities):
  """
  Turns an array of prediction probabilities into a label.
  """
  pred_indv_emotion = unique_labels[np.argmax(prediction_probabilities)]
  if 'happy' in pred_indv_emotion:
    return 'happy'
  elif 'angry' in pred_indv_emotion:
    return 'angry'
  else:
    return 'surprise'

# # get the predicted label for all the images
# for i in range(10):
#   pred_label = get_pred_label(test_segmented_predictions[i])
#   print(pred_label)


def unbatchify_test_data(data):
  '''function to unbatch test data (i.e. only images are unbatched)'''
  images = []
  for image in data.unbatch().as_numpy_iterator():
    images.append(image)
  return images

def visualize_test_predictions(prediction_prob_array, images, n = 1):

  pred_probability, image = prediction_prob_array[n], images[n]

  # Get the pred label
  pred_label = get_pred_label(pred_probability)

  # plot the image
  plt.imshow(image)
  plt.xticks([])
  plt.yticks([])

  # Change the title to - predicted label, probability of predection, and truth label
  plt.title('{} {:2.0f}% '.format(pred_label, 
                                     np.max(pred_probability)*100),
                                      color = 'red')
                                     
# unbatchify test data into images and labels for visualization
test_images = unbatchify_test_data(data = test_batched_data)


# Visualizing the test data predictions

# visualize_test_predictions(prediction_prob_array = test_segmented_predictions, 
#                            images = test_images, n = 8)

unknown_test_path = './unknown_extracted_test_images'
unknown_images_sorted_array = sorted_alphanumeric(os.listdir(unknown_test_path))

def get_Frame_ID_column(test_predictions, test_path, unknown_test_path):
  preds_test_df = pd.DataFrame(columns = ['Frame_ID', 'Emotion'])

  ordered_ID = sorted_alphanumeric([fname for fname in os.listdir(test_path)])
  preds_test_df['Frame_ID'] = ordered_ID
  preds_test_df['Emotion'] = [get_pred_label(emotion) for emotion in test_predictions]

  preds_test_df_2 = pd.DataFrame(columns = ['Frame_ID', 'Emotion'])
  ordered_unknown_ID = sorted_alphanumeric([fname for fname in os.listdir(unknown_test_path)])
  preds_test_df_2['Frame_ID'] =  ordered_unknown_ID
  preds_test_df_2['Emotion'] = 'Unknown'

  appended_preds_test_df = preds_test_df.append(preds_test_df_2)
  appended_preds_test_df_array = appended_preds_test_df.to_numpy()

  Frame_ID_series = []
  Emotion_series = []
  for row in appended_preds_test_df_array:  
    Frame_ID_series.append(row[0])
    Emotion_series.append(row[1])
  return Frame_ID_series

Frame_ID_series = get_Frame_ID_column(test_predictions = test_segmented_predictions, test_path = test_path, unknown_test_path = unknown_test_path)

# best part till now
def prepare_dataframe(test_predictions, test_path, unknown_test_path, frame_names):

  preds_test_df = pd.DataFrame(columns = ['Frame_ID', 'Emotion'])

  ordered_ID = sorted_alphanumeric([fname for fname in os.listdir(test_path)])
  preds_test_df['Frame_ID'] = ordered_ID
  preds_test_df['Emotion'] = [get_pred_label(emotion) for emotion in test_predictions]

  preds_test_df_2 = pd.DataFrame(columns = ['Frame_ID', 'Emotion'])
  ordered_unknown_ID = sorted_alphanumeric([fname for fname in os.listdir(unknown_test_path)])
  preds_test_df_2['Frame_ID'] =  ordered_unknown_ID
  preds_test_df_2['Emotion'] = 'Unknown'

  appended_preds_test_df = preds_test_df.append(preds_test_df_2)
  appended_preds_test_df.index = Frame_ID_series
  appended_preds_test_df.drop(['Frame_ID'], axis = 1, inplace = True)

  a = appended_preds_test_df.index.tolist()
  b = {}
  for x in a:
    i = (x).split('_')[0].split('e')[1]
    b[x] = int(i)

  c = sorted(a, key=lambda x: b[x])
  appended_preds_test_df = appended_preds_test_df.reindex(c)
  appended_preds_test_df['Frame_ID'] = appended_preds_test_df.index
  appended_preds_test_df.index = range(21)
  appended_preds_test_df_reorder=appended_preds_test_df.reindex(columns=['Frame_ID', 'Emotion'])
    
  return appended_preds_test_df_reorder

preds_test_df = prepare_dataframe(test_predictions = test_segmented_predictions, test_path = test_path, unknown_test_path = unknown_test_path, frame_names = Frame_ID_series)
#preds_test_df[:21]


# Saving the segmented test predictions
preds_test_df.to_csv(args.savecsv,
                     index = False)

# Get the right name of the series item for submission in correct Format
def change_df_series_items_name(df_series):
  ''' Changes the dataframe series name frame59.jpg to test59.jpg'''
  changed_arr = []
  arr = df_series.to_numpy()
  for item in arr:
    string = item.split('.')[0][0:5] 
    list1 = list(string)
    list1 = 'test'
    string = ''.join(list1)

    count = item.split('.')[0][5:]
    string2 = string + count + '.jpg'

    changed_arr.append(string2)

  return changed_arr
