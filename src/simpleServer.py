from flask import Flask, render_template, request, json

import dlib
import numpy as np
from skimage import io
import openface

app = Flask(__name__)

def facial_fuc(image):
    from keras import backend as K
    K.clear_session()

    predictor_model = 'dependency/shape_predictor_68_face_landmarks.dat'
    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    face_aligner = openface.AlignDlib(predictor_model)  # 랜드마크를 이용해 얼굴을 평면으로 만드는 친구

    # Run the HOG face detector on the image data
    detected_faces = face_detector(image, 1)

    # 모델 로드하기
    from keras.models import model_from_json
    json_file = open("result/model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # 가중치 로드하기
    loaded_model.load_weights("result/weight.h5")

    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):
        # Detected faces are returned as an object with the coordinates of the top, left, right and bottom edges
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
                                                                                face_rect.right(), face_rect.bottom()))

        # 얼굴 늘린거
        # Use openface to calculate and perform the face alignment
        alignedFace = face_aligner.align(48, image, face_rect,
                                         landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)  # 534

        alignedFace = np.dot(alignedFace[..., :3], [0.299, 0.587, 0.114])
        alignedFace = alignedFace.reshape((-1, 1, 48, 48))
        print(alignedFace.shape)
        print(alignedFace)
        output = loaded_model.predict(alignedFace)
        print("Answer :", np.argmax(output))
        return np.argmax(output)    # 한개의 얼굴만 리턴

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

