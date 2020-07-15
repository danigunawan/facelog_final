import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import cv2
import time 
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.backend import get_session
from tensorflow.python.keras.backend import clear_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
graph_mtcnn = tf.Graph()

print('=========================================================================')
with graph_mtcnn.as_default():
    sess = tf.Session(config=config)
    with sess.as_default():
        detector = MTCNN()

# detector = MTCNN()

"""
There are 3 functions:
extract_face: for extract a face from a image --> use for a image
load_faces: load faces from dirs 
extract_face_from_dataset: extract faces all dataset --> use for many images
"""

def extract_face(image,required_size=(160,160)):
    """using detect face for ROI, apply for detection from tablet 

    Args:
        image (np): frame from tablet
        required_size (tuple, optional): size for tranining. Defaults to (160,160).

    Returns:
        [np]: [face image]
    """
    t1 = time.time()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # ROI for recognition   
    # cropx = 362
    # cropy = 302
    # x,y,k = image.shape
    # startx = x//2-(cropx//2)
    # starty = y//2-(cropy//2)    
    # image = image[startx:startx+cropx,starty:starty+cropy]
    # image = np.asarray(image)
    # detect face in an image
    with graph_mtcnn.as_default():
        with sess.as_default():
            result  = detector.detect_faces(image)
            print(len(sess.graph._nodes_by_name.keys()))
    clear_session()
    # sess.close()
    # check number of faces in received image 
    if len(result) == 1:
        #extract the bouding box
        x1, y1, width, height = result[0]['box']
        #fix bug negative position
        x1 , y1 = abs(x1), abs(y1)
        x2 , y2 = x1 + width, y1 +height
        #extract the face
        image = image[y1:y2, x1:x2]
        #fit model size 
        image = cv2.resize(image,required_size)
        return image
    elif len(result)== 0:
        return np.array([])
    else:
        face = list()
        area = list()
        for i,value in enumerate(result):
            x1, y1, width, height = result[i]['box']
            area.append(width*height)
            face.append(result[i]['box'])
        x1, y1, width, height = face[area.index(max(area))]
        x1 , y1 = abs(x1), abs(y1)
        x2 , y2 = x1 + width, y1 +height
        #extract the face
        image = image[y1:y2, x1:x2]
        #fit model size 
        image = cv2.resize(image,required_size)

        return image
    print('processing time ', time.time()-t1)

def train_extract_face(image,required_size=(160,160)):
    t1 = time.time()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #detect face in an image
    with graph_mtcnn.as_default():
        with sess.as_default():
            result  = detector.detect_faces(image)
            print(len(sess.graph._nodes_by_name.keys()))
    clear_session()
    # sess.close()
    # check number of faces in received image 
    if len(result) == 1:
        #extract the bouding box
        x1, y1, width, height = result[0]['box']
        #fix bug negative position
        x1 , y1 = abs(x1), abs(y1)
        x2 , y2 = x1 + width, y1 +height
        #extract the face
        image = image[y1:y2, x1:x2]
        #fit model size 
        image = cv2.resize(image,required_size)
        return image
    elif len(result)== 0:
        return np.array([])
    else:
        face = list()
        area = list()
        for i,value in enumerate(result):
            x1, y1, width, height = result[i]['box']
            area.append(width*height)
            face.append(result[i]['box'])
        x1, y1, width, height = face[area.index(max(area))]
        x1 , y1 = abs(x1), abs(y1)
        x2 , y2 = x1 + width, y1 +height
        #extract the face
        image = image[y1:y2, x1:x2]
        #fit model size 
        image = cv2.resize(image,required_size)

        return image
    print('processing time ', time.time()-t1)



            
        
            


def load_faces(directory):
    """ Load images and extract faces for all images in a directory """
    faces = list()
    # enumerate files
    for filename in os.listdir(directory):
        # path
        path = os.path.join(directory,filename)
        # get face
        image = cv2.imread(path)
        face = train_extract_face(image)
        if face.ndim == 3:
        # store
            faces.append(face)
    return faces

def extract_face_from_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in os.listdir(directory):
        # self.labels.append(subdir)
        path = os.path.join(directory,subdir)
        # skip any files that might be in the dir
        if not os.path.isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    print('---->Extract faces from the dataset - Done!', len(X), len(y))
    return np.asarray(X), np.asarray(y) 

# if __name__ == "__main__":
#     directory = 'none'
#     image = cv2.imread('/home/dsoft/Music/me/util_facelog/facelog_v2/93878514-two-people-standing-in-poses-looking-disappointed-friends-angry-at-their-pal-because-they-waited-him.jpg')
#     a = extract_face(image)
#     print(a)