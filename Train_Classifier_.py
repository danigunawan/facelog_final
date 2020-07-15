"""
	Train SVM classifier
"""
from keras.models import load_model
import cv2
from PIL import Image
import os
import numpy as np
# from videos.utils import *
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
from sklearn.preprocessing import LabelEncoder

# HYPER-PARAMETERS
EMBEDDING_PATH = './embedding/d_soft_3classes-embeddings.npz'

class train_svm:
	def __init__(self):
		pass
	def fit(self, trainX, trainy, testX, testy):	
		# normalize input vectors
		in_encoder = Normalizer(norm='l2')
		print ("BEFORE ",trainX.shape)
		trainX = in_encoder.transform(trainX)
		print ("AFter ",trainX.shape)
		print ("Type ",type(trainX))
		testX = in_encoder.transform(testX)
		# label encode targets
		out_encoder = LabelEncoder()
		out_encoder.fit(trainy)
		trainy = out_encoder.transform(trainy)
		print ("Train Y:",trainy)
		testy = out_encoder.transform(testy)
		print('Test Y', testy)


		# save out_encoder to show the label
		pkl_filename = './models/out_encoder.pkl'
		with open(pkl_filename,'wb') as file:
			pickle.dump(out_encoder,file)

		# fit model
		print ("[INFO] Start to train the model")
		model = SVC(kernel='linear', probability=True)
		model.fit(trainX, trainy)
		# predict
		yhat_train = model.predict(trainX)
		yhat_test = model.predict(testX)
		# score
		score_train = accuracy_score(trainy, yhat_train)
		score_test = accuracy_score(testy, yhat_test)
		# save the model to disk
		pkl_filename = './models/trained_embedding_dsoft.pkl'
		with open(pkl_filename, 'wb') as file:
			pickle.dump(model, file)
		print('score of test and train :',score_test,score_train)
			
# if __name__ == "__main__":
# 	train_model_svm = train_svm()
# 	data = np.load(EMBEDDING_PATH)
# 	trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# 	train_model_svm.fit(trainX, trainy, testX, testy)