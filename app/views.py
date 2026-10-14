# Libraries
from django.shortcuts import render,redirect
from django.http import HttpResponse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier
import os

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from app.models import User

################ Home #################
def home(request):
	return render(request,'home1.html')
def login(request):
	return render(request,'loginform.html')
def loginCheck(request):
	if request.method == 'POST':
		firstname = request.POST.get('username')
		password = request.POST.get('email')
		#request.session['firstname'] = firstname 
		#print(firstname)
		print(password)
		user_object=User.objects.get(firstname=firstname,password=password)
		print('--------------------')
		print(user_object)
		try:
			print('hi')
			user_object=User.objects.get(firstname=firstname,password=password)
			print(user_object)
			if user_object is not None:
				print('hiiiiiiii')
				request.session['useremail'] = user_object.email
				return redirect('home')
		except:
			#user_object = None
			print('hello')
			return redirect('login')
	return render(request,'home.html')	
def logout(request):
	return render(request,'index.html')	
def reg(request):
	return render(request,'register.html')

######## SVM ######
def save(request):
	if request.method == 'POST':
		print('printtttttttttttttttttttttttttttttttt')
		print('checkkkkkkkkkkkkkkkkk')
		username= request.POST.get('username')
		password= request.POST.get('password')
		address= request.POST.get('address')
		email= request.POST.get('email')
		age= request.POST.get('age')
		gender= request.POST.get('gender')
		phone= request.POST.get('phone')
		user=User()
		user.firstname= request.POST.get('username')
		user.password= request.POST.get('password')
		user.address= request.POST.get('address')
		user.email= request.POST.get('email')
		user.age= request.POST.get('age')
		user.gender= request.POST.get('gender')
		user.phone= request.POST.get('phone')
		user.save()		
		return render(request,'loginform.html')
	return render(request,'loginform.html')	


######## SVM ######
def nvb(request):
	return render(request,'pacweb1.html')
def pac(request):				 
	return render(request,'result.html')
def svm(request):	
	return render(request,'acc1.html')		
def dec(request):

	return render(request,'acc1.html')
def randomf(request):
	return render(request,'acc1.html')
def mnb(request):
	return render(request,'acc1.html')
def graph(request):
	if request.method == 'POST' and request.FILES['myfile']:
		# Importing libraries
		import os
		import pandas as pd
		import numpy as np
		import matplotlib.pyplot as plt
		from matplotlib.image import imread
		import seaborn as sns
		import random
		from PIL import Image
		from sklearn.model_selection import	 train_test_split
		from tensorflow.keras.utils import to_categorical
		import tensorflow as tf
		from tensorflow.keras.models import Sequential, model_from_json
		from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D
		myfile = request.FILES['myfile']
		file1=myfile.name
		######
		
		
		##########################
		
		
		################################
		import sys,os
		print('*******************************************************************')
		print(os.path.join(os.getcwd(), 'img', file1))
		print('*******************************************************************')
		import os
		dirname = os.path.dirname(__file__)
		filename = os.path.join(dirname, myfile.name)
		print(filename)
		global model
		from keras.models import model_from_json

		dim1 = []
		dim2 = []


		

		
		#to save model into model.h5
		'''model_json = model.model.to_json()
		with open("E:/2022projects/latest/dfu/model.json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		model.model.save_weights("E:/2022projects/latest/dfu/model.h5")
		print("Saved model to disk")'''
		
		
		#to load model
		from keras.models import model_from_json
		global model
		# load json and create model
		json_file = open(r'C:\Users\LCS\Desktop\landslides\t2\model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = tf.keras.models.model_from_json(loaded_model_json)
		# load weights into new model
		model.load_weights(r'C:\Users\LCS\Desktop\landslides\t2\model.h5')
		print("Loaded model from disk")		
		
		
		
		

		# Save the uploaded file to test_images directory
		import os
		test_images_dir = r"C:\Users\LCS\Desktop\landslides\t2\test_images"
		if not os.path.exists(test_images_dir):
			os.makedirs(test_images_dir)
		
		testimg_path = os.path.join(test_images_dir, file1)
		with open(testimg_path, 'wb+') as destination:
			for chunk in myfile.chunks():
				destination.write(chunk)
		
		img = Image.open(testimg_path).convert('RGB')
		img = img.resize((50,50))
		img = np.array(img)
		im2arr = img.reshape(1,50,50,3)
		test = np.asarray(im2arr)
		test = test.astype('float32')
		test = test/255

		prediction = model.predict(test)
		y_pred1 = np.argmax(prediction, axis=1)[0]
		print(y_pred1)

		if y_pred1==0:
			import cv2
			from imutils import paths
			import imutils
			imagedisplay = cv2.imread(testimg_path)
			orig = imagedisplay.copy()
			output = imutils.resize(orig, width=400)
			cv2.putText(output, "land slide", (10, 25),	cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
			cv2.imshow("Predicted Image Result", output)
			cv2.waitKey(0)
			return render(request, 'home1.html')
		elif y_pred1==1:
			import cv2
			from imutils import paths
			import imutils
			imagedisplay = cv2.imread(testimg_path)
			orig = imagedisplay.copy()
			output = imutils.resize(orig, width=400)
			cv2.putText(output, "no landslide", (10, 25),	 cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
			cv2.imshow("Predicted Image Result", output)
			cv2.waitKey(0)
			return render(request, 'home1.html')
	return render(request,'home1.html')
def accuracy(request):
	return render(request,'index.html')
			