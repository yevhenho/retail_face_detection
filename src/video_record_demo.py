from statistics import mode

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from models.wide_resnet import WideResNet

# parameters for loading data and images

detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
age_model_path = '../trained_models/age_models/agegender_age_miniXception.hdf5'
age_model_path = '../trained_models/age-gender-estimation.hdf5'
video_path = '../data/Entrance ID camera.mp4'
# video_path = '../data/Entrance ID camera.mp4'
video_path = '../data/video (2).mp4'

emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
age_labels = get_labels('age')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
frame_window = 10
gender_offsets = (30, 60)
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)
# age_classifier = load_model(age_model_path, compile=False)
img_size = 64
age_classifier = WideResNet(img_size, depth=16, k=8)()
age_classifier.load_weights(age_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]
age_target_size = age_classifier.input_shape[1:3]

# starting lists for calculating modes
gender_window = []
emotion_window = []

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(video_path)
while (video_capture.isOpened()):

    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            rgb_face = cv2.resize(rgb_face, (gender_target_size))
            gray_face = cv2.resize(gray_face, (emotion_target_size))
            age_face = cv2.resize(rgb_face, (age_target_size))
        except:
            continue
        gray_face = preprocess_input(gray_face, False)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        rgb_face = np.expand_dims(rgb_face, 0)
        rgb_face = preprocess_input(rgb_face, False)
        gender_prediction = gender_classifier.predict(rgb_face)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]

        age_face = np.expand_dims(age_face, 0)
        age_face = preprocess_input(age_face, False)
        faces = np.empty((1, img_size, img_size, 3))
        # faces[0, :, :, :] = cv2.resize(age_face, (img_size, img_size))
        faces[0, :, :, :] = age_face
        age_prediction = age_classifier.predict(faces)
        age_text = int(age_prediction[1].dot(np.arange(0, 101).reshape(101, 1)).flatten())


        # age_label_arg = np.argmax(age_prediction)+2
        # age_label_arg = age_label_arg if age_label_arg<len(age_labels) else len(age_labels)-1
        # age_text = age_labels[age_label_arg]
        gender_window.append(gender_text+' '+str(age_text))

        if len(gender_window) > frame_window:
            emotion_window.pop(0)
            gender_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
            gender_mode = mode(gender_window)
        except:
            continue

        if gender_text == gender_labels[0]:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, gender_mode,
                  color, 0, -20, 1, 1)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
