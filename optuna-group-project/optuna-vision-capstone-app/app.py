from logging import root
from ntpath import join
import cv2
import os
import pyttsx3
import numpy as np
import torch
from torchvision.models import detection

from kivy.app import App
from kivymd.app import MDApp
from kivy.lang import Builder
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.floatlayout import MDFloatLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivymd.uix.button import MDRaisedButton, MDFillRoundFlatButton, MDIconButton
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.clock import Clock
from kivy.graphics.texture import Texture


CLASSES = open("classes.txt").read().strip().split('\n')
COLORS = np.random.uniform(0, 255, size = (len( CLASSES ), 3))

model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, progress = True,
                                                        num_classes = len(CLASSES), pretrained_backbone=True)
model.eval()


class KivyCamera(Image):

    def __init__(self, **kwargs): 
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = None

    def start(self, capture, fps=30):
        self.capture = capture
        self.event = Clock.schedule_interval(self.update, 1.0 / fps)

    def stop(self):
        Clock.unschedule(self.event)
        self.capture.release()
        self.capture = None

    def update(self, dt):
        ret, frame = self.capture.read()

        self.frame = frame
        texture = self.texture
        w, h = frame.shape[1], frame.shape[0]
        texture = Texture.create(size=(w, h))
        texture.flip_vertical()
        texture.blit_buffer(frame.tobytes(), colorfmt='bgr')
        self.texture = texture
        self.canvas.ask_update()

    def detectron(self, image):
        orig = image.copy()

        # Switch channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))

        # Process and normalize
        image = np.expand_dims(image, axis = 0) # creates a single batch of input image
        image = image / 255.0 # scales
        image = torch.FloatTensor(image)

        # Image to device
        image = image
        pred = model(image)[0]

        labels = []
        for obj in range(0, len(pred['boxes'])):
            conf = pred['scores'][obj] * 100
            
            if conf > 70:
                idx = int(pred["labels"][obj])
                labels.append(CLASSES[idx])
                bbox = pred["boxes"][obj].detach().cpu().numpy()
                (x1, y1, x2, y2) = bbox.astype('int')
                
                label = f"{CLASSES[idx]}: {conf:.2f}%"
                cv2.rectangle(orig, (x1, y1), (x2, y2), COLORS[idx], 2)
                y = y1 -15 if y1 - 15 < 15 else y1 + 15
                cv2.putText(orig, label, (x1, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        return labels


class WelcomePage(Screen):

    def screen_switch(self, instance):
        self.manager.current = 'main'
        self.manager.transition.direction = 'left'

    def dostart(self, *largs):
        global capture
        capture = cv2.VideoCapture(0)
        App.get_running_app().root.screens[1].ids.strcam.start(capture)
        

class CamStream(Screen):

    def __init__(self, **kw):
        super().__init__(**kw)
        pass

    def screen_switch(self, instance):
        self.manager.current = 'welcome'
        self.manager.transition.direction = 'right'

    def parse_tts(self):
        frame = App.get_running_app().root.screens[1].ids.strcam.frame
        label = App.get_running_app().root.screens[1].ids.strcam.detectron(frame)

        vowels = 'aeiou'
        if len(label) == 1 and label[0][0] in vowels:
            label = 'an' + label[0]
        elif len(label) == 1:
            label = 'a' + label[0]
        elif len(label) == 2:
            label = ' and '.join(label)
        else:
            label = (', ').join(label[:-1]) + ' and ' + label[-1]

        text = f'It appears you have {label} in front of you'

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()



class myapp(MDApp):
    def build(self):
        self.theme_cls.theme_style = 'Dark'
        kivyFile = Builder.load_file("build.kv")

        return kivyFile

    def on_stop(self):
        global capture
        if capture:
            capture.release()
            capture = None

if __name__ == '__main__':
    myapp().run()