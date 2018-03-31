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

# parameters for loading data and images
from src.models.wide_resnet import WideResNet

detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
age_gender_model_path = '../trained_models/age-gender-estimation.hdf5'
race_model_path = '../trained_models/racenet_vgg16_25ep_seed343.h5'
video_path = '../data/video.mp4'

emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
race_labels = get_labels('race')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
frame_window = 10
gender_offsets = (30, 60)
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
race_classifier = load_model(race_model_path, compile=False)
age_gender_classifier = WideResNet(64, depth=16, k=8)()
age_gender_classifier.load_weights(age_gender_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
age_gender_target_size = age_gender_classifier.input_shape[1:3]
race_target_size = race_classifier.input_shape[1:3]

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
            race_face = cv2.resize(gray_face, (race_target_size))
            rgb_face = cv2.resize(rgb_face, (age_gender_target_size))
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        gray_face = preprocess_input(gray_face, False)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        race_face = np.expand_dims(race_face, 0)
        race_face = np.expand_dims(race_face, -1)


        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        rgb_face = np.expand_dims(rgb_face, 0)
        age_prediction = age_gender_classifier.predict(rgb_face)
        age_text = int(age_prediction[1].dot(np.arange(0, 101).reshape(101, 1)).flatten())
        gender_text = gender_labels[np.argmax(age_prediction[0][0])]
        race_text = race_labels[np.argmax(race_classifier.predict(race_face))]
        gender_window.append(gender_text+' '+str(age_text)+':'+race_text)

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
