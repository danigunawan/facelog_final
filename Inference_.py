"""
	Predict the new image
"""

from keras.models import load_model
import keras
import cv2
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
keras.backend.clear_session()
import numpy as np
import pickle
# from videos.utils import *
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
#import from file.py
print('========----------------=============')
from extract_face import extract_face , extract_face_from_dataset
from CreateEmbedding_ import Embedding

import tensorflow as tf
from tensorflow.keras.backend import clear_session
# from custom_configs.tensorflow import *
# from custom_configs.keras import *
import time
import logging
# from mtcnn.mtcnn import MTCNN
from testAntiSpoof import AntiSpoof
ASPModel = AntiSpoof()

config=tf.ConfigProto()

EMBEDDING_MODEL_PATH = './models/trained_embedding_dsoft.pkl'
PATH_TO_CKPT = './models/frozen_inference_graph_face.pb'
pkl_filename = './models/out_encoder.pkl'

class facenet_svm():
	def __init__(self):
		self.facenet_model = Embedding()
		self.count = 0
		self.height = 160
		self.width = 160
		self.channels = 3
		with open(EMBEDDING_MODEL_PATH, 'rb') as file:
			self.embedding_model = pickle.load(file)

		# Load output_enconder from file to show the labels
		with open(pkl_filename, 'rb') as file:
			self.out_encoder = pickle.load(file)

	def predict_face(self,image):
		#detect face
		extracted_face = extract_face(image)
		
		if extracted_face.ndim == 3:
			# print('exrtraac face shape', extracted_face.shape)
			################
			ASPModel.pre_(extracted_face)
			##################
			# get face embedding 
			image_embedding = self.facenet_model.get_embedding(extracted_face)
			image_embedding = np.expand_dims(image_embedding, axis=0)
			# predict 
			proba = self.embedding_model.predict_proba(image_embedding)
			# print('proba', proba.shape)
			if np.max(proba) > 0.977:
				predict_names = self.out_encoder.inverse_transform([np.argmax(proba)])
				data = predict_names[0]
				print (f"Prediction is: {predict_names[0]}",'\tProbability: %.6f' %(np.max(proba)*100))
			else:
				predict_names = self.out_encoder.inverse_transform([np.argmax(proba)])
				print (f"Prediction is: {predict_names[0]}",'\tProbability: %.6f' %(np.max(proba)*100))			
				data = "Unknown"
				print("Unknown faces.")
		else:
			image_embedding = 'None'
			data = "Unknown"
			print('No face in the image!')
		return data, image_embedding


if __name__ == "__main__":
	f = facenet_svm()
	directory = './train_dir'
	assert len(os.listdir(directory)) != 0
	for subdir in os.listdir(directory):
		# self.labels.append(subdir)
		path = os.path.join(directory,subdir)
		# skip any files that might be in the dir
		if not os.path.isdir(path):
			continue
		print('---------> ',subdir)
		for filename in os.listdir(path):
			# path
			path_image = os.path.join(path,filename)
			# get face
			image = cv2.imread(path_image)
			f.predict_face(image)
