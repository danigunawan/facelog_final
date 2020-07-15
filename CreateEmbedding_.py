import numpy as np
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from keras.models import load_model
from sklearn.model_selection import train_test_split
from extract_face import extract_face, extract_face_from_dataset
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.backend import get_session
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.models import load_model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
graph_facenet = tf.Graph()
with graph_facenet.as_default():
    sess1 = tf.Session(config=config)
            # sess = tf.Session(config=config)
# from custom_configs.keras import *
# from custom_configs.tensorflow import *
# HYPER-PARAMETERS

DATASET_NAME = 'D-dataset'
MODEL_PATH = './models/facenet_keras.h5'

class Embedding(object):
    def __init__(self):
        with graph_facenet.as_default():

            with sess1.as_default():
                # set_session(sess)
                # sess = get_session()
                # init = tf.compat.v1.global_variables_initializer()
                # sess1.run(init)
                self.model = load_model(MODEL_PATH, compile = False)
                print(len(sess1.graph._nodes_by_name.keys()))

        # self.detector = MTCNN()
        print ("[INFO] Model input:",self.model.inputs)
        print ("[INFO] Model output:",self.model.outputs)

    def get_embedding(self,extracted_face):
        """ get the face embedding for one face """
        # scale pixel values
        extracted_face = extracted_face.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = extracted_face.mean(), extracted_face.std()
        extracted_face = (extracted_face - mean) / std
        # transform face into one sample
        extracted_face = np.expand_dims(extracted_face, axis=0)
        # make prediction to get embedding
        with graph_facenet.as_default():
            with sess1.as_default():
                yhat = self.model.predict(extracted_face)
        
                # print(len(sess.graph._nodes_by_name.keys()))
        clear_session()
        return yhat[0]
    def create_embedding(self, train_dir = './train_dir'):
        #  # Load images from path and split train & test
               
        #-----The first time
        X,y = extract_face_from_dataset(train_dir)
        trainX, testX, trainy, testy = train_test_split(X,y, test_size=0.2, random_state=42)
        np.savez_compressed(f'./embedding/{DATASET_NAME}.npz', trainX, trainy, testX,testy)
        
        # -----The second time
        split_images = np.load(f'./embedding/{DATASET_NAME}.npz')
        trainX, trainy, testX, testy = split_images['arr_0'] ,split_images['arr_1'] ,split_images['arr_2'] ,split_images['arr_3'] 
        print('-->Load face.npz, Done\ntrain: {}, label: {}, test: {}, label: {}'.format(trainX.shape,trainy.shape, testX.shape, testy.shape))
        # convert each face in the train set to an embedding
        newTrainX = list()
        for face_pixels in trainX:
            embedding = self.get_embedding(face_pixels)
            newTrainX.append(embedding)
        newTrainX = np.asarray(newTrainX)
        print(newTrainX.shape)
        # convert each face in the test set to an embedding
        newTestX = list()
        for face_pixels in testX:
            embedding = self.get_embedding(face_pixels)
        # print (embedding)
            newTestX.append(embedding)
        newTestX = np.asarray(newTestX)
        print(newTestX.shape)
        # save arrays to one file in compressed format
        np.savez_compressed(f'./embedding/{DATASET_NAME}-embeddings.npz', newTrainX, trainy, newTestX, testy)
        print('--{0}--{1}--{2}--{3}--> Create Embeddings - Done!'.format(newTrainX.shape, trainy.shape, newTestX.shape, testy.shape ))
        print('number class of train {}, test {}'.format(np.unique(trainy).shape, np.unique(testy).shape))
        return newTrainX, trainy, newTestX, testy

    def embedding_follow_name(self,X,y,name):
        name_Embeddings = X[name == y]
        return name_Embeddings

# if __name__ == "__main__":
#     eb = Embedding()
#     trainX, trainy, testX, testy = eb.create_embedding()
    
#     name_train = np.unique(trainy)
#     for name in name_train:
#         name_Embeddings_1 = eb.embedding_follow_name(trainX,trainy,name)
#         name_Embeddings_2 = eb.embedding_follow_name(testX,testy,name)
#         name_Embeddings = np.r_[name_Embeddings_1,name_Embeddings_2]
#         #add put function 
#         print(name,name_Embeddings.shape)
        