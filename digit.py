from colorama import Fore, Back, Style
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfile
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
from tkinter import messagebox
import tensorflow.python.keras.engine
import os


def statement(stmnt):
    print(Fore.YELLOW, Back.LIGHTBLACK_EX, Style.BRIGHT, stmnt, Style.RESET_ALL)

# model1 = load_model('./Digit Identification/tkinter_model')
# model = load_model('./Digit Identification/model')

# model1 = load_model('tkinter_model')
model = load_model('model')

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
        return preds
    
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
        return pred

def open_file(speed):
    file = askopenfile(mode ='r', filetypes =[
        ('Video', '*.mp4'),
        ('Pictures', '*.png')
    ])
    if file is not None:
        ls = predict(model,file.name,speed)
        if file.name.split('.')[-1] in ['png']:
            messagebox.showinfo('success'.title(),f'the model identified the given image as {ls}'.title())
        else:
            stmnt = '''the model identified the digits in the given video as - 
            '''.title()
            for i in ls:
                stmnt+=str(i)+' '*3
            messagebox.showinfo('success'.title(),stmnt)

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
        
                
def predict_digit(file):
    # resize image to 28x28 pixels
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # txt_off_x = 5
    # txt_off_y = gray.shape[0]-25
    # box = ((txt_off_x,txt_off_y), (txt_off_x+text_width+1,txt_off_y-text_height-5))

    img = tf.keras.utils.normalize(img)
    img = np.array(img).reshape(-1, 28, 28, 1)
    res = model.predict(img)[0]
    #     status = np.argmax(pred)

    # img = img.resize((28,28))
    # #convert rgb to grayscale
    # img = img.convert('L')
    # img = np.array(img)
    # #reshaping to support our model input and normalizing
    # img = img.reshape(1,28,28,1)
    # img = img/255
    # #predicting the class
    # res = model.predict([img])[0]
    os.remove(file)
    return np.argmax(res), max(res)


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        self.title('Identify Single Handwritten Digit')
        
        self.speed_var = tk.IntVar(value=1)
        self.canvas = tk.Canvas(self, width=400, height=400, bg="black", cursor="dotbox")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Identify", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)
        self.canvas.grid(row=0, column=0, pady=1, )  # sticky=W, )
        self.canvas.old_coords = None
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.speed = Entry(self, textvariable=self.speed_var).place(x=410,y=450)
        self.button_open = tk.Button(self, text='open a picture or video instead'.title(), command=lambda:open_file(self.speed_var.get()) ).grid( row=3,column=0,pady=10, padx=2)
        self.label1 = Label(self, text='Video Frame Skips').place(x=300,y=450)
        self.button_cam = Button(self,text='Open Camera',command=lambda:open_cam()).grid( row=4,columnspan=3,pady=10, padx=2)


    
    
    
    def clear_all(self):
        self.canvas.delete("all")
        self.canvas.old_coords=None
        


    def classify_handwriting(self):
        id_ = self.canvas.winfo_id()
        x_ = self.canvas.winfo_x()
        y_ = self.canvas.winfo_y()
        wid = self.canvas.winfo_width()
        hei = self.canvas.winfo_height()
        wx = self.winfo_rootx()
        wy = self.winfo_rooty()
        # print(id_)
        # print(x_,y_)
        # print(wx,wy)
        # print(wid,hei)
        x = wx+x_
        y = wy+y_
        x1 = x+wid
        y1 = y+hei
        file_name = 'img_.png'
        ImageGrab.grab().crop((x+50,y+50,x1+120,y1+120)).save(file_name)
        
        # HWND = self.canvas.winfo_id()  
        # rect = win32gui.GetWindowRect(HWND)  
        # im = ImageGrab.grab(rect)
        # file_name = 'img.png'
        
        # im.save(file_name)
        digit, acc = predict_digit(file_name)
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        # self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='white')

        # x, y = event.x, event.y
        if self.canvas.old_coords:
            x1, y1 = self.canvas.old_coords
            self.canvas.create_line(self.x, self.y, x1, y1, width=5, fill='white')
        self.canvas.old_coords = self.x, self.y
        
    
app = App()
mainloop()