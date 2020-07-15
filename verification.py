'''
    This project is based on two research papers.
    Citations{
        @article{
            1604.02878,
            Author = {Kaipeng Zhang and Zhanpeng Zhang and Zhifeng Li and Yu Qiao},
            Title = {Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks},
            Year = {2016},
            Eprint = {1604.02878},
            Eprinttype = {arXiv},
            Doi = {10.1109/LSP.2016.2603342},
        }

        @article{
            1503.03832,
            Author = {Florian Schroff and Dmitry Kalenichenko and James Philbin},
            Title = {FaceNet: A Unified Embedding for Face Recognition and Clustering},
            Year = {2015},
            Eprint = {1503.03832},
            Eprinttype = {arXiv},
            Doi = {10.1109/CVPR.2015.7298682},
        }
    }
    @Author: MD Sarfarazul Haque.

    Used two pre-trained model in this project, one is for face detection and
    another is for face verificaton.

    FaceNet and MTCNN are the two models.

    For more information about these two models give a shot to following git-repo:
        FaceNet: https://github.com/nyoki-mtl/keras-facenet.git
        MTCNN: https://github.com/ipazc/mtcnn.git

'''


import cv2
import numpy as np 
from build_db import build_DB
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import pickle
import os.path
from keras import backend as K
import tensorflow as tf
import time 
print("loading model") 
model = load_model('models/facenet_keras.h5', compile = False)
model._make_predict_function()
# graph = tf.get_default_graph()
print("done loading model")
path = 'data/feature_db.pickle'

def load_data(path):
        '''
            This function loads data into X and Y array respectively
            X: This array contains features extracted from training samples
            Y: This array contains labels associated with those training samples
        '''

        # To check is there si a database of training sample present or not
        if not os.path.exists(path):
            # If database is not present then we have to create one
            db = build_DB()
            db.make_db()

        # Load tha database into our program.
        with open(path, 'rb') as f:
            data_dict = pickle.load(f)
        print('Loading data!')
        # Set the training data to self.X and self.Y
        global X
        global Y
        X, Y = data_dict['features'], data_dict['labels']
        return X, Y
        print('Data loaded:')
load_data(path)


class face_recog(object):
    # Main class of the project this class deals with the face recognition part of the project.
    def __init__(self):
        '''
            This function initializes all the class parameters.
        '''
              # Getting the object of build_DB class to deal with database.
        # self.b_db = build_DB()
        # Loading the FaceNet model.
        # star_time = time.time()
        # print('loading model ======================')
        # # self.model = load_model('models/facenet_keras.h5',compile=False)
        # # self.model._make_predict_function()
        # stop_time = time.time()
        # total_time = stop_time - star_time
        # print('total time run loadmodel',total_time)
       
        # print('model has been loaded ===============================')
        # Setting the path of database file.
        # self.p_file = 'data/feature_db.pickle'
        # Setting shape of each image.
        self.height = 160
        self.width = 160
        self.channels = 3
        
        # Getting the object of MTCNN() class to detect a face in an image.
        self.detector = MTCNN()

        # Setting the opencv parameters.
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.color = (0, 255, 255)
        self.line = cv2.LINE_AA

        # Setting the threshold value.
        self.threshold = 15.0
    def loadModel(self):
        # load the pre-trained Keras model (here we are using a model
        # pre-trained on ImageNet and provided by Keras, but you can
        # substitute in your own networks just as easily)
        global model
        model = load_model('models/facenet_keras.h5',compile = False)
    
        print('load model has done')

    def get_euclidean(self, X, Y):

        return np.sqrt(np.sum(np.square(np.subtract(X, Y))))

    # def load_data(self):
    #     '''
    #         This function loads data into X and Y array respectively
    #         X: This array contains features extracted from training samples
    #         Y: This array contains labels associated with those training samples
    #     '''

    #     # To check is there si a database of training sample present or not
    #     if not os.path.exists(self.p_file):
    #         # If database is not present then we have to create one
    #         db = build_DB()
    #         db.make_db()

    #     # Load tha database into our program.
    #     with open(self.p_file, 'rb') as f:
    #         data_dict = pickle.load(f)
    #     print('Loading data!')
    #     # Set the training data to self.X and self.Y
    #     self.X, self.Y = data_dict['features'], data_dict['labels']
    #     print('Data loaded:')



        
    def recog_each_face(self, face):
        global model
        '''
            This function recognize a single face, extracted from a single frame.

            @param1: face: The face extracted from the frame
                    it's shape is (self.height, self.width, self.channel)

            @return1: It returns the label predicted.
        '''
        # Reshaping the array into batch format so FaceNet model can work.
        # Normalizing the data to reduce computing time and memory.
        face = face.astype(np.float16).reshape((1, self.height, self.width, self.channels))
        face /= 255.0
        # Extracting feature vector.
        print('bug final start')
        print(type(face))
        global model
        global X
        global Y
        # print('gia tri X', X)
        feature = model.predict(face)
        # K.clear_session()

        print('bug final end')
        dist = []
        # Calculating euclidean distance.
        for s_x in X:
            dist.append(self.get_euclidean(s_x, feature))
        
        dist = np.array(dist)
        # Getting the most similar face.
        indx = np.argmin(dist)
        print(indx)

        if dist[indx] < self.threshold:
            return Y[indx]
        else:
            return "Opps!"
    def extracting(self, face):
        global model
        '''
            This function recognize a single face, extracted from a single frame.

            @param1: face: The face what will be extracted from the frame
                    it's shape is (self.height, self.width, self.channel)

            @return1: It returns the label predicted.
        '''
        # x, y, w, h = face['box']
        # # Getting Region Of Image(ROI)
        # f_img = frame[y:y+h, x:x+w]
        # Resizing the face in the shape of (self.width, self.height)
        # f_img = cv2.resize(face, (self.width, self.height))
        # Reshaping the array into batch format so FaceNet model can work.
        # face = np.asarray(face)
        face = face.astype(np.float16).reshape((1, self.height, self.width, self.channels))
        # Normalizing the data to reduce computing time and memory.
        face /= 255.0
        # Extracting feature vector.
        # print('bug final start')
        # print(type(face))
        global model
        global X
        global Y
        # print('gia tri X', X)
        feature = model.predict(face)
        return feature

    def embedded_facenet(self, frame):
        """[summary]

        Arguments:
            frame {[type]} -- [description]

        Returns:
            [] -- [description]
        """
        faces = self.detector.detect_faces(frame)
        try:
            len(faces) == 1
        except:
            print("image not suitable for training")
            
        if len(faces) != 1:
            print("image not suitable for training")
        else:
            for face in faces:
            # Getting the co-ordinates of the bounding box.
                x, y, w, h = faces['box']
                # Getting Region Of Image(ROI)
                f_img = frame[y:y+h, x:x+w]
                # Resizing the face in the shape of (self.width, self.height)
                f_img = cv2.resize(f_img, (self.width, self.height))
                # Calling the helper function to get the label.
                print('---------------------------------')
                # embeddedface = self.extracting(self,f_img)

                # Reshaping the array into batch format so FaceNet model can work.
                face = face.astype(np.float16).reshape((1, self.height, self.width, self.channels))
                # Normalizing the data to reduce computing time and memory.
                face /= 255.0
                # Extracting feature vector.
                # print('bug final start')
                # print(type(face))
                global model
                global X
                global Y
                # print('gia tri X', X)
                feature = model.predict(face)
                # return feature
                
                print("in thu label :", feature.shape)
                # Drawing rectangle and putting text on the bounding box of each fce
                # cv2.rectangle(frame, (x,y), (x+w, y+h), self.color, 2, self.line)
                # cv2.putText(frame, label, (x-3, y-3), self.font, 0.5, self.color, 1)
        return feature





    def recog_each_frame(self, frame):
        '''
            This function deals with each frame and extract the faces from current frame and 
            sends those faces to self.recog_each_face(self, face) to generate label.

            @param1: frame: The frame currently being operated for face verification.

            @return1: It returns the frame with labeled associated with each face.
        '''

        # Uses MTCNN library to detect faces
        # For more information about this function below give a check to following git-repository
        # https://github.com/ipazc/mtcnn.git
        faces = self.detector.detect_faces(frame)

        # This for loop draws bounding box aroung the face with a name associated with the box.
        # If there is no face present in the frame then this function returns
        #  the original frame passed to it.
        print('detect face was be doneeeeeeeeeeeee')
        for face in faces:
            # Getting the co-ordinates of the bounding box.
            x, y, w, h = face['box']
            # Getting Region Of Image(ROI)
            f_img = frame[y:y+h, x:x+w]
            # Resizing the face in the shape of (self.width, self.height)
            f_img = cv2.resize(f_img, (self.width, self.height))
            # Calling the helper function to get the label.
            print('---------------------------------')
            label = self.recog_each_face(f_img)
            print("in thu label :", label.shape)
            # Drawing rectangle and putting text on the bounding box of each fce
            # cv2.rectangle(frame, (x,y), (x+w, y+h), self.color, 2, self.line)
            # cv2.putText(frame, label, (x-3, y-3), self.font, 0.5, self.color, 1)
        return label

    def test(self, mode, img, save_path=None):
        '''
            This is the driver function of this class.
            This function read the image or video content either from Webcam or File
            and sends each frame to self.recog_each_frame(self, frame) to get faces recognized
            and show each frame through cv2.imshow().

            @param1: mode: Whether the source is image or video
                    values are, `image`, `video`.

            @param2: path: Path to the file in case of video file or image mode or 
                            webcam number in case webcam mode. 

            @param3: save_path: If save_path != None and mode == `image` then save the output to
                            location specified by save_path.
        '''
        # self.load_data()
        #  mode selected is 'image'
        if mode == 'image':
            # check image file...
            if len(img) == 0:
                print('No image found!')
                return
            ID1 = self.recog_each_frame(img)
            print("label is :", ID1)
        return ID1

# face = face_recog()
# img = cv2.imread('dev/test/2.png')
# face.test('image',img)
