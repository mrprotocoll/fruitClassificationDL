import os
from crypt import methods
from tkinter.tix import Tree
from turtle import width
from urllib import request
from flask import Flask, render_template, request, send_from_directory
from object_detector import *

from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
classes = [['Green Apple',"Fresh"],['Orange','Fresh'],['Red Apple',"Fresh"],['Apple','Bad'],['Orange','Bad'],['Not Orange or Apple','-']]
# classes = [['Orange','Spoilt'],['Orange','Fresh'],['Green Apple',"Fresh"],['Apple','Spoilt'],['Red Apple',"Fresh"],['Not Orange or Apple','-']]
def sized(px):
    diameter = (px*2)/37
    if diameter <= 5:
        return "Small"
    elif(diameter > 5 and px < 7):
        return "Medium"
    else:
        return "Large"
@app.route('/', methods=['POST','GET'])
def index():
    # labels = os.listdir("../dataset/test/")
    # print(labels)
    
    if request.method == 'POST':
        target = os.path.join(APP_ROOT, 'static/images/')
        # target = os.path.join(APP_ROOT, 'static/')
        print(target)
        if not os.path.isdir(target):
                os.mkdir(target)
        print(request.files.getlist("file"))
        predictions = []
        filenames = []
        size = []
        for upload in request.files.getlist("file"):
            # print(upload)
            # print("{} is the file name".format(upload.filename))
            filename = upload.filename
            destination = "/".join([target, filename])
            # print ("Accept incoming file:", filename)
            # print ("Save it to:", destination)
            upload.save(destination)

            #import tensorflow as tf
            import numpy as np
            from tensorflow.keras.preprocessing import image
            from keras.models import load_model

            new_model = load_model('final_model.h5')
            #new_model.summary()
            test_image = image.load_img('static/images/'+filename,target_size=(50,50))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = new_model.predict(test_image)
            # print("result----------------------")
            # print(result)
            result1 = result[0]
            isfruit = False
            for i in range(5):
                # print("result-------1-----------")
                # print(result1[i])
                if result1[i] == 1.:
                    isfruit = True
                    break;
            prediction = classes[i] if isfruit else classes[5]
            predictions.append(prediction)
            filenames.append(filename)
            # print(predictions)
            # print(filenames)
            # ----------------

            # calculate image size
            imge = cv2.imread(destination,1)
            orig = imge.copy()
            imge = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)   #converting the color to grayscale
            imge = cv2.GaussianBlur(imge, (21, 21), cv2.BORDER_DEFAULT)
            circles = cv2.HoughCircles(imge, cv2.HOUGH_GRADIENT, 0.9, 120,
              param1=30,
              param2=20,
              minRadius=10,
              maxRadius=0)
            all_cirlcles = np.uint16(np.around(circles))
            #Creating reference box
            radius = 0
            for i in all_cirlcles[0,:]:
                if radius < i[2]: radius = i[2]
            size.append(sized(radius))


        # return send_from_directory("images", filename, as_attachment=True)
        if len(size) > 0:
            return render_template("result.html",image_names=filenames, texts=predictions,counts = len(predictions),sizes = size)
        else:
            return render_template("result.html",image_names=filenames, texts=predictions,counts = len(predictions),sizes = [["nil","nil"]])
    else:
        return render_template("index.html")

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("static/images", filename)

if __name__ == "__main__":
    app.run(debug=True)
