from flask import Flask, request, redirect, render_template, send_file, send_from_directory, safe_join, abort
import os
import cv2
import numpy as np


# Initialize the Flask application
app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "/home/cidacoder/pythonAPI"

# Function takes image in b/w(gray) and original image(frame)
# and return image with detector rectangles.
def detect(gray, frame):
    
    # Loading the cascades
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('Mouth.xml')
    nose_cascade = cv2.CascadeClassifier('Nose.xml')

    alpha=0.9
    beta=50

    gray=cv2.addWeighted(gray,alpha,np.zeros(gray.shape, gray.dtype),0,beta)
    
    frameCopy = frame.copy()

    #face
    #parameters of functions (GreyImage, ScalingFactorOfImage, NoOfNeighbourZoneFacesNearThisFace)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faceCount=0
    for (x, y, w, h) in faces:
        faceCount = faceCount + 1
        #parameters (OrginalImage,(TOpLefCorner), (botthomRightCorner),(RGB),BorderWidth)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        

        #Rectangular Area of Face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        f=0
        g=0

        #Mouth (find Mouth only inside the rectangular area of face)
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.7, 20)
        for (sx, sy, sw, sh) in mouth:
            f=1
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    
    
        #eyes (find eyes only inside the rectangular area of face)
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
        #Nose (find nose only inside the rectangular area of face)
        nose = nose_cascade.detectMultiScale(roi_gray, 1.1, 20)
        for (nx, ny, nw, nh) in nose:
            g=1
            cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (255, 255, 0), 2)

        if(f==0 and g==0):
            print('Mask Laga hai')
            cv2.rectangle(frameCopy, (x, y), (x+w, y+h), (0, 255, 0), 4)
            #cv2.putText(frameCopy,'Mask Found',(x+50,y-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),5, cv2.LINE_AA)
        else:
            print('Corona Aaa jayega!!!')
            cv2.rectangle(frameCopy, (x, y), (x+w, y+h), (0, 0, 255), 4)
            #cv2.putText(frameCopy,'Mask NOT Found',(x+50,y-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),5, cv2.LINE_AA)
    if(faceCount == 0):
        print('No face found')

    return frameCopy


def maskPredict(img):

    frame = cv2.imread(img)
    #convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #repaint with apropriate rectangles
    canvas = detect(gray, frame)
    cv2.imwrite('predict.jpg', canvas)



@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":

        if request.files:

            image = request.files["image"]

            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            print(image.filename)
            maskPredict(image.filename)

            address = str(request.url).split('//')[1]
            address = address.split('/')[0]
            address = 'http://'+address+'/get-image/predict.jpg'
            print(address)
            #redirect to show image
            return redirect(address)

    return render_template("upload_image.html")

@app.route("/get-image/<image_name>")
def get_image(image_name):
    #set 'as_attachment' = True if want to download directly
    try:
        return send_from_directory(app.config["IMAGE_UPLOADS"], filename=image_name, as_attachment=False)
    except FileNotFoundError:
        abort(404)

app.run(host="0.0.0.0", port=5000)