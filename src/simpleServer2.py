from flask import Flask, render_template, request, json
import cv2
import numpy as np
from skimage import io

app = Flask(__name__)

def facial_fuc(image):
    from keras import backend as K
    K.clear_session()

    # path = '../lib/python3.6/site-packages/cv2/data/'
    path = 'dependency/'
    face_cascade = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(grayImage, 1.1, 3)

    # 모델 로드하기
    from keras.models import model_from_json
    json_file = open("result/model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # 가중치 로드하기
    loaded_model.load_weights("result/weight.h5")

    print("Number of faces detected: " + str(faces.shape[0]))
    for (x, y, w, h) in faces:
        new_image = grayImage[y:y + h, x:x + w]

        new_image = cv2.resize(new_image, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
        new_image = new_image.reshape((-1, 1, 48, 48))

        output = loaded_model.predict(new_image)
        print("Answer :", np.argmax(output))
        return np.argmax(output)    # 한개의 얼굴만 리턴.

@app.route("/")
def hello():
    return render_template('index.html')

@app.route('/fileUpload', methods=['POST'])
def fileUpload():
    file = request.files['file'];

    result = facial_fuc(io.imread(file))
    return json.dumps({'status': str(result)});

if __name__ == "__main__":
    app.run()

