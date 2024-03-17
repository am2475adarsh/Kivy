import pickle
import cv2
import mediapipe as mp
import numpy as np

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.graphics import Color, RoundedRectangle
from kivy.core.window import Window

# Set background color
Window.clearcolor = (0.1, 0.1, 0.1, 1)

# Load the model
model_dict = pickle.load(open('./model42.p', 'rb'))
model = model_dict['model']

# Initialize mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the labels dictionary
labels_dict = {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z', 'space': 'space'}

class CameraWidget(Image):
    def __init__(self, **kwargs):
        super(CameraWidget, self).__init__(**kwargs)
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # Update at 30 fps

    def update(self, dt):
        ret, frame = app.cap.read()

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            while len(data_aux) < 84:
                data_aux.append(0)

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]

            if predicted_character in labels_dict.values():
                predicted_label = next(key for key, value in labels_dict.items() if value == predicted_character)
            else:
                predicted_label = -1

            if predicted_label != -1:
                predicted_character = labels_dict[predicted_label]
            else:
                predicted_character = 'Unknown'

            app.prediction_label.text = predicted_character

        buf1 = cv2.flip(frame, 0)
        buf = buf1.tobytes()  # Use tobytes() instead of tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.texture = texture1

class HandGestureRecognitionApp(App):
    def build(self):
        self.cap = cv2.VideoCapture(0)
        
        root_layout = BoxLayout(orientation='vertical')
        
        # Top bar
        top_bar = BoxLayout(size_hint=(1, None), height=50, padding=[10, 0, 10, 0])
        with top_bar.canvas.before:
            Color(1, 1, 1, 1)  # Dark gray background color for the navbar
            RoundedRectangle(size=top_bar.size, pos=top_bar.pos, radius=[0, 0, 15, 15])  # Rounded corners for the navbar
        top_bar.add_widget(Label(text='SignSpeak', color=(1, 1, 1, 1), font_size='24sp'))
        root_layout.add_widget(top_bar)

        camera_layout = BoxLayout(orientation='vertical')
        # camera_layout.add_widget(Label(text='Camera Feed', size_hint=(1, 0.1)))
        camera_widget = CameraWidget(size_hint=(1, 0.9))
        camera_layout.add_widget(camera_widget)

        prediction_box = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        prediction_box.add_widget(Label(text='Prediction: ', color=(1, 1, 1, 1), font_size='14sp'))
        self.prediction_label = Label(text='-', font_size=20, color=(1, 1, 1, 1))
        
        with prediction_box.canvas.before:
            Color(0.15, 0.15, 0.15, 1)  # Darker gray background color for the prediction box
            self.prediction_box_rect = RoundedRectangle(size=prediction_box.size, pos=prediction_box.pos, radius=[15, 15, 15, 15])  # Rounded corners for the prediction box
            
        prediction_box.bind(size=self._update_rect, pos=self._update_rect)
        prediction_box.add_widget(self.prediction_label)
        camera_layout.add_widget(prediction_box)

        root_layout.add_widget(camera_layout)

        return root_layout

    def _update_rect(self, instance, value):
        self.prediction_box_rect.size = instance.size
        self.prediction_box_rect.pos = instance.pos

if __name__ == '__main__':
    app = HandGestureRecognitionApp()
    app.run()
