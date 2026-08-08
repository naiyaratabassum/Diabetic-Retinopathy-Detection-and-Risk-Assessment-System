import numpy as np 
import os 
from random import shuffle 
from tqdm import \
    tqdm  
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for,render_template_string, request
import sqlite3
import cv2
import shutil
import requests
from osmapi import OsmApi
from datetime import datetime
from werkzeug.utils import secure_filename
import uuid




app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

app = Flask(__name__)


@app.route('/')
def index():
    print(app.url_map)  
    return render_template('index.html')


@app.route('/hospitals_map')
def hospitals_map():
    print("Hospitals Map Route Accessed")
    return render_template('hospitals_map.html')

@app.route('/dashboard.html')
def dashboard():
    print("dashboard Accessed")
    return render_template('dashboard.html')

@app.route('/patient.html')
def patient():
    print("patient analyser page Accessed")
    return render_template('patient.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/userlog.html')
def demo():
    return render_template('userlog.html')

@app.route('/compare')
def compare_reports():
    return render_template('compare.html') 




@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        
                
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        

        shutil.copy("test/"+fileName, dst)
        image = cv2.imread("test/"+fileName)
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        
        edges = cv2.Canny(image, 100, 200)
        cv2.imwrite('static/edges.jpg', edges)
        
        retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)
        
        kernel_sharpening = np.array([[-1,-1,-1],
                                    [-1, 9,-1],
                                    [-1,-1,-1]])

        
        sharpened = cv2.filter2D(image, -1, kernel_sharpening)

        
        cv2.imwrite('static/sharpened.jpg', sharpened)

        
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'diabeticRetinopathy-{}-{}.model'.format(LR, '2conv-basic')
    
        def process_verify_data():
            verifying_data = []
            for img in os.listdir(verify_dir):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])
                np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        

        
        tf.compat.v1.reset_default_graph()
        

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 5, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        diseasename=" "
        rem=" "
        rem1=" "
        str_label=" "
        accuracy=""
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))
            conf=model_out[np.argmax(model_out)]
            print(conf)
            
            
            if np.argmax(model_out) == 0:
                str_label = 'Mild'
                print("The predicted image of the Mild is with a accuracy of {} %".format(model_out[0]*100))
                accuracy="The predicted image of the Mild is with a accuracy of {}%".format(model_out[0]*100)
                rem = "The remedies for Mild Retinopathy are:\n\n "
                rem1 =["Track Sugar Levels Regularly",  
                "Follow-Up A Proper Diet.", 
                "Regular Consultancy"]
                
            elif np.argmax(model_out) == 1:
                str_label = 'Moderate'
                print("The predicted image of the Moderate is with a accuracy of {} %".format(model_out[1]*100))
                accuracy="The predicted image of the Moderate is with a accuracy of {}%".format(model_out[1]*100)
                rem = "The remedies for Diabetic Retinopathy are: "
                rem1 = ["Laser Treatement.",
                "Surgical Removal Of Vitreous Gel.", 
                "Anti-Vascular Endothelial Growth Factor.", 
                "Anti-Inflamatory Medicine"]
                
            elif np.argmax(model_out) == 2:
                str_label = 'Normal'
                print("The predicted image of the Normal is with a accuracy of {} %".format(model_out[2]*100))
                accuracy="The predicted image of the Normal is with a accuracy of {}%".format(model_out[2]*100)

            elif np.argmax(model_out) == 3:
                str_label = 'proliferate'
                print("The predicted image of the proliferate is with a accuracy of {} %".format(model_out[3]*100))
                accuracy="The predicted image of the proliferate is with a accuracy of {}%".format(model_out[3]*100)
                
                rem = "The remedies for Diabetic Retinopathy are: "
                rem1 = ["Injecting Medication Into Eye.",
                "Photocoagulation.", 
                "Panretinal Photocoagulation", 
                "Vitrectoimy."]
            elif np.argmax(model_out) == 4:
                str_label = 'Severe'
                print("The predicted image of the Severe is with a accuracy of {} %".format(model_out[4]*100))
                accuracy="The predicted image of the Severe is with a accuracy of {}%".format(model_out[4]*100)
                
                rem = "The remedies for Diabetic Retinopathy are: "
                rem1 = ["Injecting Medication Into Eye.",
                "Photocoagulation.", 
                "Panretinal Photocoagulation", 
                "Vitrectoimy."]
            
            

           
                        
               
            A=float(model_out[0])
            B=float(model_out[1])
            C=float(model_out[2])
            D=float(model_out[3])
            E=float(model_out[4])
            0
            dic={'Mild':A,'Moderate':B,'Normal':C,'proliferate':D,'Severe':E}
            algm = list(dic.keys()) 
            accu = list(dic.values()) 
            fig = plt.figure(figsize = (5, 5))  
            plt.bar(algm, accu, color ='maroon', width = 0.3)  
            plt.xlabel("Comparision") 
            plt.ylabel("Accuracy Level") 
            plt.title("Accuracy Comparision between Diabetic Retinopathy....")
            plt.savefig('static/matrix.png')

            
                         

        return render_template('results.html', status=str_label,accuracy=accuracy, remedie=rem, remedie1=rem1, ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg",ImageDisplay4="http://127.0.0.1:5000/static/sharpened.jpg",ImageDisplay5="http://127.0.0.1:5000/static/matrix.png")
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
