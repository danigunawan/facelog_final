import sys
import os
import cv2
# import face_recognition
import math
import time
from sklearn import neighbors, svm
import os.path
import pickle
# from face_recognition.face_recognition_cli import image_files_in_folder
from threading import Thread
import mongo_embedded
# import face_detecedon
import tensorflow as tf
# tf.compat.v1.disable_v2_behavior 
import numpy as np
import collections
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from datetime import datetime
from backend_request_util import DsoftHttpRequest 
from verification import face_recog
from mtcnn.mtcnn import MTCNN
backend = DsoftHttpRequest()
PATH_TO_CKPT = 'models/face_detection/frozen_inference_graph_face.pb'
#Variables
trained_knn_model_path = 'models/output/trained_mtcnn_knn_dsoft_model.clf'
trained_knn_model_path_dlib = 'models/output/trained_mtcnn_knn_dsoft_model_dlib.clf'
train_dir = 'train_dir'
distance_threshold = 0.42
n_neighbors = 15


# time = 0


class MTCNN_KNN():
    def __init__(self):
        # logging.basicConfig(format="[ MTCNN_KNN %(levelname)s ] %(messages)s", level=logging.INFO, stream=sys.stdout)
        # self.log = logging.getLogger()
        self.count = 0
        self.height = 160
        self.width = 160
        self.channels = 3
        self.detector = MTCNN()
        self.model = load_model('models/facenet_keras.h5',compile = False)
        self.model._make_predict_function()
        print('model has been load--------------------------')
        # if not os.path.exists(trained_knn_model_path):
        #     self._create_update_trained_model(train_dir)
        
        # self.knn_clf, self.class_name, self.emb_array= self._load_knn_clf_model()
       
      
        
    def run(self,frame):

        
        
        if self.count == 0:
           
        # setup tensorflow graph
            
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
            # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            # self.model = load_model('liveness_dsoft.model')
            # configuration for possible GPU use
            # tf.compat.v1.disable_v2_behavior
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            
            detection_graph = tf.Graph()

            # load frozen tensorflow detection model and initialize 
            # the tensorflow graph
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

                self.sess = tf.Session(graph=detection_graph, config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
                self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                self.boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                self.scores =detection_graph.get_tensor_by_name('detection_scores:0')
                self.classes = detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_detections =detection_graph.get_tensor_by_name('num_detections:0')
            self.count = 1
        
        image_expanded = np.expand_dims(frame, axis=0)
        (boxes, scores, classes, num_detections) = self.sess.run(
            [self.boxes, self.scores, self.classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})
        
        boxes=np.squeeze(boxes)
        classes =np.squeeze(classes)
        scores = np.squeeze(scores)

        h,w,_ = frame.shape
        face_locations = []
        area_array = []
        max_face_roi = []
        for i in range(len(boxes)):
            if scores[i] > 0.5:
                
                # box[1],box[0],box[3],box[2]
                # left,top,right, bottom
                top,left,bottom,right = boxes[i]
                top = int(top*h)
                bottom = int(bottom*h)
                left = int(left*w)
                right = int(right*w)
                h_face = bottom - top
                w_face = right - left 
                print(top,right,bottom,left)
                if h_face > 0 and w_face > 0:  
                    area_array.append(h_face*w_face) 
                    face_locations.append((top,right,bottom,left))
                # if h_face > 25 and w_face > 25:                 
                #     if left > w/6 and right < 5*w/6 and top > h/6 and bottom < 5*h/6: 
                #         area_array.append(h_face*w_face) 
                #         face_locations.append((top,right,bottom,left))
        if len(area_array) > 0:
            np.asarray(area_array,dtype='int')
            max_area = np.argmax(area_array)
            max_face_roi.append(face_locations[max_area])
            return max_face_roi       
        return []                #     if left > w/6 and right < 5*w/6 and top > h/6 and bottom < 5*h/6: 
                #         area_array.append(h_face*w_face) 
                #         face_locations.append((top,right,bottom,left))
        if len(area_array) > 0:
            np.asarray(area_array,dtype='int')
            max_area = np.argmax(area_array)
            max_face_roi.append(face_locations[max_area])
            return max_face_roi       
        return []
   
    def get_knn_model(self):
        return self.knn_clf
    def training(self,X, Y):
        print(len(X) , len(Y))
        # clf = svm.SVC(kerneer='linearn')
        # clf.fit(X,Y)

        knn_clf = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, weights='distance')
        knn_clf.fit(X,Y)
        with open(trained_knn_model_path, 'wb') as f:
            pickle.dump((knn_clf,Y,X), f)
        print('Trained MTCNN_KNN Completed') 

    def train_newstaff(self, name, image):
            """train model and put embedded to mongoDB

            Arguments:
                name {str} -- name of users
                image {np} -- file images

            Returns:
                make response -- put id, embedded, class to embedded collection

            """
            X = np.array([])
            Y = []

            _id = backend.get_user_id(name)
            
            if _id is not None:
                return "User not exsting..."
            else:

                # image = face_recognition.load_image_file(image)
                face_bounding_boxes = self.run(image)
                # print(class_dir)

                if len(face_bounding_boxes) != 1:
                    # print("this image has 2 face")
                    print("Image {} not suitable for training: {}".format(image, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
                else:
                    image = image[:, :, :: -1]
                    k = face_recognition.face_encodings(image, known_face_locations = face_bounding_boxes)[0]
                    print("check embedded",k)
                    # h = np.array([2,3])
                    X = np.append(X, k, axis=None)
                    print("gia tri k",X.shape)
                    Y.append(name)

                embedded = [{
                    "embedded": np.random.random((100,100)),
                    "class_user": "thanh",
                }]
                return X
                # collection.insert_one(embedded)
        
    def train_firstmodel(self, train_dir):
        # self.log.error("Model not found! Creating new model")
        # Loop through each person in the trainning set
        '''
        Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
        '''
    
        X = np.array([])
        Y = []
        for class_dir in os.listdir(train_dir):
            
            # self.log.info('Training {}'.format(class_dir))
            if not os.path.isdir(os.path.join(train_dir, class_dir)):
                continue
            #Loop through each training image for the current person
            for image_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
                image = face_recognition.load_image_file(image_path)
              
                face_bounding_boxes = self.run(image)
                print(class_dir)

                if len(face_bounding_boxes) != 1:
                    # print("this image has 2 face")
                    print("Image {} not suitable for training: {}".format(image_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
                else:
                    image = image[:, :, :: -1]
                    # X.append(face_recognition.face_encodings(image, known_face_locations= face_bounding_boxes)[0])
                    X = np.append(X,face_recognition.face_encodings(image, known_face_locations = face_bounding_boxes)[0],axis = None)
                    
                    # Y.append(class_dir) 
                    print("value of x demention", len(X))
            print("value of X", X, len(X))
            

            print("value of Y",class_dir)
            
            mongo_embedded.add_staff(X, class_dir)
            X = np.array([])
            print("X sau khi xu ly", X)    

    def create_update_trained_model(self, train_dir):
        # self.log.error("Model not found! Creating new model")
        # Loop through each person in the trainning set
        '''
        Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
        '''
    
        X = []
        Y = []
        X1 = []
        for class_dir in os.listdir(train_dir):
            
            # self.log.info('Training {}'.format(class_dir))
            if not os.path.isdir(os.path.join(train_dir, class_dir)):
                continue
            #Loop through each training image for the current person
            for image_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
                image = face_recognition.load_image_file(image_path)
                # face encoding with facenet and detect face using MTCNN
                faces = self.detector.detect_faces(image)
                if len(faces)==1:
                    for face in faces:
                    
                    # Getting the co-ordinates of the bounding box.
                        x, y, w, h = face['box']
                        print('-----------',x, y, w, h)
                        # Getting Region Of Image(ROI)
                        if y == 0 :
                            break
                        if x == 0 :
                            break
                        f_img = image[abs(y):y+h, x:x+w]
                        print(f_img.shape)
                        # Resizing the face in the shape of (self.width, self.height)
                        f_img = cv2.resize(f_img, (160, 160))
                        # Calling the helper function to get the label.
                        face = f_img.astype(np.float16).reshape((1, self.height, self.width, self.channels))
                        face /= 255.0
                        # Extracting feature vector.
                        faces_encodings = self.model.predict(face)
                        X.append(faces_encodings[0])
                        Y.append(class_dir)

                #     # label = self.recog_each_face(f_img)
                #     # print("in thu label :", label.shape)

                #------------face encoding with Dlib-------------

                # face_bounding_boxes = self.run(image)
              
                # if len(face_bounding_boxes) != 1:
                #     # print("this image has 2 face")
                #     print("Image {} not suitable for training: {}".format(image_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
                # else:
                #     t00 = time.time()
                #     print("time to test",t00)
                #     image = image[:, :, :: -1]
                #     top, right, bottom, left = face_bounding_boxes[0]
                #     image_save = image[top:bottom, left:right , :]
                #     f_img = cv2.resize(image_save, (self.width, self.height))
                #     facenet embedded
                #     f_img = np.asarray(f_img)
                #     faces_encodings = face_recog.extracting(self, f_img)
                #     print(faces_encodings.shape)
                #     X.append(faces_encodings[0])



                    # X.append(face_recognition.face_encodings(image, known_face_locations= face_bounding_boxes)[0])
                    # Y.append(class_dir)   
            
        #Create and train the KNN classifier
   
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors = 15,p = 2, weights='distance')
        knn_clf.fit(X,Y)
        with open(trained_knn_model_path, 'wb') as f:
            pickle.dump((knn_clf,Y,X), f)
        print('Trained MTCNN_KNN Completed') 

        # --------dlib embedded-------------------

        # knn_clf_dlib = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, weights='distance')
        # knn_clf_dlib.fit(X,Y)
        # with open(trained_knn_model_path_dlib, 'wb') as f:
        #     pickle.dump((knn_clf_dlib,Y,X), f)
        # print('Trained MTCNN_KNN Completed') 


    def _load_knn_clf_model(self):
        ''' Load face clasification model '''
        # self.log.info('Load MTCNN_KNN model') 
        with open(trained_knn_model_path, 'rb') as f:
            # knn_clf, classY, list_encoding = pickle.load(f)
            knn_clf, classY, classX = pickle.load(f)
        return knn_clf, classY, classX

    def get_predict(self, frame):
        
        self.knn_clf, self.class_name, self.emb_array= self._load_knn_clf_model()
        predict = []
        # select ROI included face 
        path = '/home/dsoft/Desktop/Project/Facelog_v2/enjoyworks/ai/dsoft_face_recognition/face'
        cv2.imwrite(os.path.join(path , 'image_roi.jpg') , frame  )
        cropx = 362
        cropy = 302
        x,y,k = frame.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        frame = frame[startx:startx+cropx,starty:starty+cropy]
        # Get face locations in the frame
        frame = frame[:,:,::-1]
        x1, h1, k = frame.shape



        face_bounding_boxes = self.run(frame)
        print('len of bouding box', len(face_bounding_boxes))
        if len(face_bounding_boxes) == 0:
            return [None,None,False]
        print('x, h of frame', x1, h1)
        print('bounding box', face_bounding_boxes)

        faces_encodings = face_recognition.face_encodings(frame, known_face_locations = face_bounding_boxes)
        
        closest_distances = self.knn_clf.kneighbors(faces_encodings, n_neighbors = 5)
        # Use for filter the noise value ( more than 2 value are matches with above )
        names_array = []
        
        for i in range(len(face_bounding_boxes)):
           
            list_name = []
            print(closest_distances)
            
            for j,find_name in enumerate(closest_distances[1][i]):
                if closest_distances[0][i][j] > 0.45:
                    list_name.append('Unknown')
                else:                   
                    list_name.append(self.class_name[find_name])
            print(list_name)
            counts = collections.Counter(list_name)
            name_in_list = sorted(counts, key=lambda x: -counts[x])
            name_appear_most = list_name.count(name_in_list[0]) 
            if name_appear_most/len(list_name) > 0.59 and len(name_in_list) < 3:
                name = name_in_list[0]
                predict.append('97.20127')
            else:
                name = 'Unknown'
                predict.append('45.16212')
            names_array.append(name)
        print(names_array)
        
        return [names_array[0],predict[0],True]
 

    def _load_knn_clf_model_dlib(self):
            ''' Load face clasification model '''
            # self.log.info('Load MTCNN_KNN model') 
            with open(trained_knn_model_path_dlib, 'rb') as f:
                # knn_clf, classY, list_encoding = pickle.load(f)
                knn_clf_dlib, classY, classX = pickle.load(f)
            return knn_clf_dlib, classY, classX

    def get_predict_dlib(self, frame):
        
        self.knn_clf_dlib, self.class_name, self.emb_array= self._load_knn_clf_model_dlib()
        predict = []
        #get ROI of image contain face
        cropx = 362
        cropy = 302
        # path = '/home/dsoft/Desktop/dsoft_face_recognition/final/image_roi'
        shapeimage = frame.shape
        x,y,k = frame.shape
        # name = 'image_fit_shape'
        # print('x :', x, 'y :', y)
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        frame = frame[startx:startx+cropx,starty:starty+cropy]
        # cv2.imwrite(os.path.join(path , 'image_roi.jpg') , frame)
        

        # print('shape of face image', frame.shape)
        # Get face locations in the frame  
        face_bounding_boxes = self.run(frame)
        print('len of face_bounding box ', len(face_bounding_boxes))
        if len(face_bounding_boxes) == 0:
            return [None,None,False]
        
        #replace embedded face using facenet
        # top, right, bottom, left = face_bounding_boxes[0]
        # image_save = frame[top:bottom, left:right , :]
        # f_img = cv2.resize(image_save, (self.width, self.height))
        # faces_encodings = face_recog.extracting(self, f_img)
        # print(faces_encodings.shape)
   
        faces_encodings = face_recognition.face_encodings(frame, known_face_locations = face_bounding_boxes)
        # faces_encodings =np.asarray(faces_encodings)
        # print('shape of face_encodings',faces_encodings.shape) 
        # Use the KNN model to find the best matches for the frame
        # print()
        closest_distances = self.knn_clf_dlib.kneighbors(faces_encodings, n_neighbors = 5)
        # Use for filter the noise value ( more than 2 value are matches with above )
        names_array = []
        # result = self.knn_clf_dlib.predict(faces_encodings)
        # print('predicted result', result)
        # test = ["Triqn"]
        # test = np.asarray(test)
        # from sklearn.metrics import accuracy_score
        # print("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(test, result)))
        # print("score of dlib ", self.knn_clf_dlib.predict_proba(faces_encodings))
        
        for i in range(len(face_bounding_boxes)):
           
            list_name = []
            print(closest_distances)
            
            for j,find_name in enumerate(closest_distances[1][i]):
                if closest_distances[0][i][j] > 0.45:
                    list_name.append('Unknown')
                else:                   
                    list_name.append(self.class_name[find_name])
            print(list_name)
            counts = collections.Counter(list_name)
            name_in_list = sorted(counts, key=lambda x: -counts[x])
            name_appear_most = list_name.count(name_in_list[0]) 
            if name_appear_most/len(list_name) > 0.59 and len(name_in_list) < 3:
                name = name_in_list[0] 
                predict.append('97.20127')
            else:
                name = 'Unknown'
                predict.append('45.16212')
            names_array.append(name)
        print(names_array)
        
        return [names_array[0],predict[0],True] 
