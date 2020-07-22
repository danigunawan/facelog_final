import sys
import urllib
import threading
import numpy as np
import numpy
import cv2
import time
import datetime
import requests
from flask import Flask, render_template, Response, send_file, request, json, make_response, jsonify,redirect, url_for
# from Dsoft_Knn_Face_Recognition import MTCNN_KNN
# mtcnn = MTCNN_KNN()
from multiprocessing import Process, Manager, cpu_count, Queue
import multiprocessing
import logging
import api_constant as api
from DsoftUtitlities import Utitlities
# from person_detection import PersonDetector
# from camera import CameraThread
# import people_counter
# import face_recognition_core
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# import send_data
import psutil
from flask_cors import CORS, cross_origin
import mongodb_cameras
from backend_request_util import DsoftHttpRequest
import mongo_embedded
import calendar

from build_db import build_DB
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import pickle
import os.path
# from verification import face_recog 
# import face_recognition
from extract_face import extract_face, extract_face_from_dataset
from Inference_ import facenet_svm 
from CreateEmbedding_ import Embedding
from Train_Classifier_ import train_svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

emb_model = Embedding()
svm_model = facenet_svm()


# from mtcnn.mtcnn import MTCNN



'''pid in ubuntu'''
pid_in = None
pid_out = None
pid_mainprocess = None


queue_pid = Queue()
connection_camera_in = Queue(1)
connection_camera_out = Queue(1)

cam_1 = mongodb_cameras.get_camera('in')
print('camera ',cam_1)
cam_2 = mongodb_cameras.get_camera('out')
# cam_1 = api.PI_CAM1
# cam_2 = api.PI_CAM2
cam_top = api.CAMERA_TOP

Logging = None
Utils = None
Utils = Utitlities()
request_backend = DsoftHttpRequest()

''' Output camera '''
output_cam_in = Queue(1)
output_cam_out = Queue(1)

output_cam_top = Queue(1)

''' Input camera '''
input_cam_in = Queue(1)
input_cam_out = Queue(1)
input_cam_top = Queue(1)

''' Send data to tablet'''
data_tablet_in = Queue(1)
data_tablet_out = Queue(1)

''' information camera '''
ip_camera = Queue(1)
name_camera = Queue(1)

bytes = bytes()
bytes_out = bytes

app = Flask(__name__)
CORS(app)
app.config["RECOGNITE_IMAGES"] = 'recogniteImage'
app.config["MANUAL_OPEN"] ='manualOpen'
#initial global variable

FRAME_OUT = None
FRAME_IN = None
FRAME_TOP = None

global saveImage
global directory
global isEnroll
global model_MTCNN
global outWrite
# graph = tf.get_default_graph()


@app.route('/started', methods=['GET'])
def is_servered():
    return Response(status=200)

@app.before_first_request
def start_cameras():
    """ setup AI models, load server before run routers """
    global Logging, pid_mainprocess, pid_in,pid_out
    Logging.info("Login to Dsoft-Backend") 

    # thread_input()   #running input
    # thread_output()  #running output

    ''' get pid of processes are running '''
    while not queue_pid.empty():
        data = queue_pid.get()
        if list(data.keys())[0] == 'In':
            pid_in = data['In']
        else:
            pid_out = data['Out']
    pid_mainprocess = multiprocessing.current_process().pid
def auto_start_app():
    def start_loop():
        global Logging
        # Log
        logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
        Logging = logging.getLogger()
        not_started = True
        Logging.info("Starting flask server...")
        while not_started:
            try:
                r = requests.get(api.AI_TRIGGER_START_FLASK)
                if r.status_code == 200:
                    not_started = False
                    Logging.info("Started flask server!")
            except:
                Logging.error("The Flask server not start yet, keep trying...")
            time.sleep(2)
    thread = threading.Thread(target=start_loop)
    thread.start()  


@app.route("/opendoor", methods = ["POST"])
def opendoor():
    url = 'http://192.168.3.104/open'
    company = request.form.get('company',type=str)
    floor = request.form.get('floor',type=str)
    print(company, floor)
    jsondata = {"name": company,"floor": floor}
    jon = {"Name": company,"Floor": floor}
    data1 = request_backend.opendoor(url, jon)
    res = "OK"

    if res in data1:
        res = {"message": "the door are opened", "code": 200,"error": "false", "data":jsondata }
    else:
        res = {"message": "the door are not opened", "code": 404,"error": "True", "data":jsondata }

    return make_response(jsonify(res), 200)

@app.route("/distance_recog", methods = ["POST"])
def distance():
    """idont nkow what purpose of this api created by me

    Returns:
        NONE: NONE
    """

    # get encoding from DB
    print("error")
    
    username, all_embeded = mongo_embedded.get_full_embedded()
    all_embeded = all_embeded.reshape(-1,128)
    # print('username-----------',username)
    # distance = np.sqrt((all_embeded -));
    image = request.files['image']
    nparr = np.fromstring(image.read(), np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    # extracting face image to encoding face
    face  = extract_face(img_np)
    if len(face) == 0:
        data = {"message": "image are not suitabel", "code": 404,"error": "True", "data":staffname }

    else:
        starttime = time.time()
        encoding_face = svm_model.extract_embedding(face)    
        distance = np.sort(np.sqrt(np.sum((all_embeded - encoding_face)**2,axis=1)), axis=0)
        print('-------------------',len(distance))
        print('time of processing', time.time() - starttime)
                
    res = {"message": "", "code": 404,"error": "True", "data":'sthing' }
    return make_response(jsonify(res), 200)



@app.route("/add_newstaff", methods = ["POST"])
def training():
    name = request.args.get('staffname',type=str)
    # image =request.files["image"]
    images = request.files.getlist('image')
    embedding_face = list()
    for i, image in enumerate(images):
        nparr = np.fromstring(image.read(), np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
        extracted_face = extract_face(img_np)
        print('face shape', face.shape)
        if extracted_face.ndim == 0 :
            data = {"message": "image are not suitabel", "code": 404,"error": "True", "data": name }
        else:
            data = {"message": "image has been extract to embedding and push DB", "code": 200,"error": "False", "data": name }
            emb = emb_model.get_embedding(face)
            face_embedding.append(emb)
        return face_embedding
    # put new staff embedding into mongodb 
    mongo_embedded.add_staff(face_embedding,staffname)
    # get all embedding from mongodb
    username, all_embeded = mongo_embedded.get_full_embedded()
    all_embeded = all_embeded.reshape(-1,128)
    # split data for svm training  
    trainX, testX, trainy, testy = train_test_split(all_embeded,username, test_size=0.2, random_state=42)
    train_svm_model.fit(trainX,trainy, testX,testy)
    data = {"message": "train has been done", "code": 200,"error": "False", "data": 'first train' }
    return make_response(jsonify(data), 200)

@app.route("/training", methods = ['POST'])
def training_svm():
    '''    # the first trainining
        # choose one of two options args 
        #  [1] options = 'firsttraining' - first training with initial dataset when you apply system for customer 
        #  [2] options = 'addstaff' - add new staff 
        #   if options = 'training':'''
    train_svm_model = train_svm()
    trainX, trainy, testX, testy = emb_model.create_embedding()
    #put data into DB
    # name_train = numpy.unique(trainy)
    # for name in name_train:
    #     name_Embeddings_1 = emb_model.embedding_follow_name(trainX,trainy,name)
    #     name_Embeddings_2 = emb_model.embedding_follow_name(testX,testy,name)
    #     name_Embeddings = numpy.r_[name_Embeddings_1,name_Embeddings_2]
    #     print(name,name_Embeddings.shape)
    #     mongo_embedded.add_staff(name_Embeddings, name)
    #training 
    train_svm_model.fit(trainX,trainy, testX,testy)
    data = {"message": "train has been done", "code": 200,"error": "False", "data": 'first train' }
    return make_response(jsonify(data), 200)

@app.route('/recognizing', methods=["POST", "GET"])
def recognizing():
    """recognition with two threshold state 
    [1]: SVM with threshold probabitity = 0.97
    [2]: Face Verify with threshhold distance = 0.7, the rusult from state [1] will be used to for state [2]
    Returns:
        [json]]: [using for request]
    """

    t1 = time.time()
    direction = request.args.get('direction', type=str)
    image = request.files["image"]
    nparr = np.fromstring(image.read(), np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    data, face_embedding = svm_model.predict_face(img_np)
    print("RECOGNITING=======================================")
    print(data)
    time.time()
    name = data
    if name != None:
        if name != "Unknown" and face_embedding != 'None':
            userInfo = request_backend.get_user_info(name.lower())
            if userInfo != None:
                # Distance state
                print('check docker run')
                emb,count = mongo_embedded.get_embedded(data)
                print(emb.shape)
                emb = emb.reshape(-1, 128)
                tolerance = 0.9
                # normalizer embedded face 
                normalizer = Normalizer(norm='l2')
                face_embedding = normalizer.transform(face_embedding)
                emb = normalizer.transform(emb)
                # calculate distance 
                result = np.linalg.norm(emb - face_embedding, axis=1) <= tolerance
                print('+++++++++++++',result)
                # result = face_recognition.compare_faces(emb,encoding_face,tolerance=0.45)
                result = np.asarray(result)
                proba = np.sum(result[result==True])/result.shape[0]
                print('-------------------------------------------------',proba)
                if proba >= 0.5:
                    resRecog = {"name": name, "staff": userInfo}
                    res = {"message": "Staff are indentified", "code": 200, "error": False, "data": resRecog}
                    print(res)
                    return make_response(jsonify(res), 200)
                else:
                    resRecog = {"name": name, "staff": userInfo}
                    res = {"message": "cant trust the result !!!!", "code": 304, "error": False, "data": resRecog}     
                    print(res)               
            else:
                resRecog = {"name": name, "staff": None}
                res = {"message": "Person are indentified but not register in database", "code": 304, "error": False, "data": resRecog}
                print(res)
                return make_response(jsonify(res), 200)
        else:
            resRecog = {"name": name, "staff": None}
            res = {"message": "Person are not indentified", "code": 404, "error": False, "data": resRecog}
            print(res)
            t2 = time.time()
            print('--------')
            print('-------------',t2)
            t3 = t2 - t1
            print('time of processing',t3)
            return make_response(jsonify(res), 200)
            print('-*8-------------')
    res = {"message": "Failed to recognize image", "code": 300, "error": True, "data": None}
    print(res)

    return make_response(jsonify(res), 200)

@app.route('/verify',methods = ["POST","GET"])
def verify():
    # get frame and username from tablet 
    staffname = str(request.args.get("username"))
    image = request.files["image"]
    print('username-----------',staffname)
    nparr = np.fromstring(image.read(), np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    # extracting face image to encoding face
    extracted_face = extract_face(img_np)
    # check exist face after face detection
    if extracted_face.ndim ==3:
        face_embedding = emb_model.get_embedding(extracted_face)
        face_embedding = np.expand_dims(face_embedding, axis=0)
        emb,count = mongo_embedded.get_embedded(staffname)
        # reshape for caculate distance 
        emb = emb.reshape(-1, 128)
        tolerance = 0.8
        # Normalize face embedding
        normalizer = Normalizer(norm='l2')
        face_embedding = normalizer.transform(face_embedding)
        emb = normalizer.transform(emb)
        # calculate distance 
        print(emb.shape)
        result = np.linalg.norm(emb - face_embedding, axis=1) <= tolerance
        # result = face_recognition.compare_faces(emb,encoding_face,tolerance=0.45)
        print(result)
        result = np.asarray(result)
        proba = np.sum(result[result==True])/result.shape[0]
        print(proba)
        if proba >= 0.5:
            data = {"message": "id has been verified", "code": 200,"error": "False", "data":staffname }

        else:
            data = {"message": "id has not   verified", "code": 404,"error": "True", "data":staffname }
    else:   
        data = {"message": "image are not suitabel", "code": 404,"error": "True", "data":staffname }
    return make_response(jsonify(data), 200)

    
@app.route('/postworklog', methods=["POST"])
def postWorklog():
    direction = request.args.get('direction', type=str)
    name = request.args.get('username', type=str)
    image = request.files["image"]
    nparr = np.fromstring(image.read(), np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    access = ""
    if direction == "true":
        access = "In"
    else:
        access = "Out"
    image_evidence = Utils.capture_evidence(img_np, name, access, False)
    if name == "Unknown":
        code = request_backend.post_worklog(image_evidence, name, direction)
        if access == "In":
            request_backend.notify_stranger(image_evidence)
            res = {"message": "Stranger detect! Sent alarm notification", "code": 200, "error": False, "data": None}
            return make_response(jsonify(res), 200)
        else:
            res = {"message": "Stranger detect!", "code": 200, "error": False, "data": None}
            return make_response(jsonify(res), 200)
    else:
        code = request_backend.post_worklog(image_evidence, name, direction)
        if code == 200:
            userInfo = request_backend.get_user_info(name)
            res = {"message": "Post work log successful", "code": code, "error": False, "data": userInfo}
            return make_response(jsonify(res), 200)
        else:
            res = {"message": "Post work log error", "code": code, "error": True, "data": None}
            return make_response(jsonify(res), 200)

@app.route('/alert', methods=["POST"])
def alert():
    image = request.files["image"]
    nparr = np.fromstring(image.read(), np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    image_evidence = Utils.capture_evidence(img_np, "Unknown", "In", False)
    code = request_backend.notify_stranger(image_evidence)
    res = {"message": "Stranger detect!", "code": code, "error": False, "data": None}
    return make_response(jsonify(res), 200)

@app.route('/getUserInfo', methods=["GET"])
def get_user_info():
    user_name = request.args.get('user', type=str)
    data = request_backend.get_user_info(user_name)
    if data != None:
        resRecog = {"name": user_name, "staff": data}
        res = {"message": "Post work log successful", "code": 200, "error": False, "data": resRecog}
        return make_response(jsonify(res), 200)
    else:
        res = {"message": "Staff not exist or wrong username", "code": 404, "error": True, "data": None}
        return make_response(jsonify(res), 200)
    res = {"message": "Falied to get staff information", "code": 300, "error": True, "data": None}
    return make_response(jsonify(res), 200)

if __name__ == '__main__':
    auto_start_app()
    global saveImage
    global directory
    global isEnroll
    global model_MTCNN
    global outWrite
    model_MTCNN = None
    isEnroll = False
    saveImage = False
    directory = None
    outWrite = None

    app.run(host=api.AI_FLASK_HOST, port= api.AI_FLASK_PORT, debug=False)
