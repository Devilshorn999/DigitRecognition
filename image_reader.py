from colorama import Fore, Back, Style
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import messagebox


'''
This is for running on command line.
'''

def statement(stmnt):
    print(Fore.YELLOW, Back.LIGHTBLACK_EX, Style.BRIGHT, stmnt, Style.RESET_ALL)
    
model1 = load_model('./Digit Identification/tkinter_model')
model = load_model('./Digit Identification/model')

parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
parser.add_argument('-m', '-model', help='Select your model. for model with 2 dense layer - 1 , for model with 1 dense layer - 2'.title(), dest='model', type=int, default=1, choices=[1,2])
parser.add_argument('-f', '-file', help='file path of the image or video'.title(), dest='file_path', type=str, default='./myImage/2022-02-03-12-20-54.mp4')
parser.add_argument('-s', '-speed', help='Frame Read Speed'.title(), dest='frame_speed', type=int, default=1)
parser.add_argument('-w', '-cam', help='open web cam'.title(), dest='webcam', type=int, default=0, choices=[0,1])
args = parser.parse_args()

print('-' * 30)
print('Parameters')
print('-' * 30)
for key, value in vars(args).items():
    print('{:<20} := {}'.format(key, value))
print('-' * 30)

def predict(model, file, frame_speed):
    preds = []
    ext = file.split('.')[-1]
    if ext.lower() in 'mp4,ts,webm,MPG,MP2,MPEG,MPE,MPV,ogg,M4P,M4V,avi,wmv,mov,qt,flv,swf,avchd,gif'.lower().split(','):
        font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        size = 1.5

        vid = cv2.VideoCapture(file)

        if not vid.isOpened():
            vid = cv2.VideoCapture(0)
        if not vid.isOpened():
            raise IOError('File Not Open')

        text = 'some text in a box!'.title()
        text_width,text_height=cv2.getTextSize(text,font,fontScale=size,thickness=1)[0]


        counter = 0
        while True:
            ret, frame = vid.read()
            counter+=1;
            if (counter%frame_speed)==0:
                try:
                    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                except:
                    break
                resized = cv2.resize(gray,(28,28),interpolation=cv2.INTER_AREA)

                # txt_off_x = 5
                # txt_off_y = gray.shape[0]-25
                # box = ((txt_off_x,txt_off_y), (txt_off_x+text_width+1,txt_off_y-text_height-5))

                img = tf.keras.utils.normalize(resized)
                img = np.array(img).reshape(-1,28,28,1)
                pred = model.predict(img)
                status = np.argmax(pred)
                # print(status)
                # print(type(status))
                preds.append(status)
                x1,y1,w1,h1 = 0,0,175,175
                cv2.rectangle(frame,(x1,x1),(x1+100,y1+100),(0,0,0),-1)
                cv2.putText(frame,status.astype(str),(x1+int(w1/5),y1+int(h1/2)),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,.7,(0,0,255),2)

                cv2.imshow('handwritten digits recognition',frame)

                if cv2.waitKey(2)&0xFF==ord('q'):
                    break
        vid.release()
        cv2.destroyAllWindows()
    
    elif ext.lower() in 'png,jpeg,tiff,psd,xcf,ai,cdr,tif,bmp,jpg,eps,raw,cr2,nef,orf,sr2'.split(','):
        img = cv2.imread(file)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
        img = tf.keras.utils.normalize(img)
        img = np.array(img).reshape(-1,28,28,1)
        pred = model.predict(img)
        print('the model identified the digit in the image as'.title(),end='')
        pred = np.argmax(pred)
        statement(np.argmax(pred))
        # messagebox.showinfo("showinfo", f'The Digit in the picture is {pred}')

def open_cam():
    print('Initializing cam')
    for i in range(4):
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            cam.set(3,640)
            cam.set(4,480)
            while True:
                ret, frame = cam.read()
                img = np.asarray(frame)
                img = cv2.resize(img,(280,280))
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                # cv2.imshow('video',img)
                img = cv2.equalizeHist(img)
                img = tf.keras.utils.normalize(img)
                img = cv2.resize(img,(28,28))
                # cv2.imshow('video1',img)
                img = img.reshape(1,28,28,1)
                prob = model.predict(img)
                class_ = np.argmax(prob)
                prob = round(prob[0][class_]*100,2)
                try:
                    print(class_,'-',prob)
                    if prob>30:
                        cv2.putText(frame,
                                    str(class_) + ' ' + str(prob),
                                    (5,30),
                                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                                   1,(0,255,0),2)
                    cv2.putText(frame,
                                ' '*63+'Press q to exit',
                                (5,20),
                                cv2.FONT_ITALIC,
                               0.5,(0,0,255),1)
                except IndexError:
                    continue
                cv2.imshow('      Camera', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cam.release()
        else:
            stmnt = f'No Camera connected'.title()
            root = tk.Tk()
            root.withdraw()
            message = messagebox.showinfo('Error'.title(),stmnt)
            root.destroy()
            continue
        stmnt = '''
            Exiting Camera.
            '''
        root = tk.Tk()
        root.withdraw()
        message = messagebox.showinfo('success'.title(),stmnt)
        root.destroy()
        cv2.destroyAllWindows()
        break


if args.model == 1:
    statement('2 dense layer model enabled')
    if args.webcam == 0:
        predict(model, args.file_path, args.frame_speed)
    else:
        open_cam()
else:
    statement('2 dense layer model enabled')
    if args.webcam == 0:
        predict(model, args.file_path, args.frame_speed)
    else:
        open_cam()