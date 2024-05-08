# -*- coding: utf-8 -*-
 

import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
from tkinter import messagebox

window = tk.Tk()

window.title("Face Recognition unlocking system")

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'

 

# window.configure(background='green')

# Open and convert the image to a format compatible with Tkinter

image = Image.open("rec.jpg")
image = image.resize((window.winfo_screenwidth(), window.winfo_screenheight()), Image.ANTIALIAS)
background_image = ImageTk.PhotoImage(image)

# Create a Canvas widget
canvas = tk.Canvas(window, width=image.width, height=image.height)
canvas.pack()

# Add the background image to the Canvas
canvas.create_image(0, 0, anchor=tk.NW, image=background_image)



window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)





message = tk.Label(window, text="Face Recognition unlocking system", fg="black", font=('times', 30, ' bold '), bd=0, highlightthickness=0)

message.place(x=500, y=60)

lbl = tk.Label(window, text="Enter ID",width=20  ,height=2  ,fg="black"   ,font=('times', 15, ' bold ') ) 
lbl.place(x=400, y=200)

txt = tk.Entry(window,width=20  ,fg="black",font=('times', 15, ' bold '))
txt.place(x=700, y=215)

lbl2 = tk.Label(window, text="Enter Name",width=20  ,fg="black"     ,height=2 ,font=('times', 15, ' bold ')) 
lbl2.place(x=400, y=300)

txt2 = tk.Entry(window,width=20  ,fg="black",font=('times', 15, ' bold ')  )
txt2.place(x=700, y=315)

lbl3 = tk.Label(window, text="Notification : ",width=20  ,fg="black"   ,height=2 ,font=('times', 15, ' bold underline ')) 
lbl3.place(x=400, y=400)

message = tk.Label(window, text=""   ,fg="black"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
message.place(x=700, y=400)

lbl3 = tk.Label(window, text="Notification : ",width=20  ,fg="black"    ,height=2 ,font=('times', 15, ' bold  underline')) 
lbl3.place(x=400, y=650)


message2 = tk.Label(window, text="" ,fg="black"   ,activeforeground = "green",width=30  ,height=2  ,font=('times', 15, ' bold ')) 
message2.place(x=700, y=650)

lbl4= tk.Label(window, text="Notification : ",width=20  ,fg="black"   ,height=2 ,font=('times', 15, ' bold  underline')) 
lbl4.place(x=400, y=750)


message3= tk.Label(window, text="" ,fg="black"   ,activeforeground = "green",width=30  ,height=2  ,font=('times', 15, ' bold ')) 
message3.place(x=700, y=750)
 
# def clear():
#     txt.delete(0, 'end')    
#     res = ""
#     message.configure(text= res)

# def clear2():
#     txt2.delete(0, 'end')    
#     res = ""
#     message.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def carownerimage():        
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(y,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage/ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                # Inside the carownerimage() function, just before cv2.imwrite()
                print("Image dimensions (x, y, w, h):", x, y, w, h)
                print("Gray image dimensions:", gray.shape)

                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        with open('carownerdetails\carownerdetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)
    
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImage\Trainner.yml")
    res = "Image Trained"#+",".join(str(f) for f in Id)
    message.configure(text= res)

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    print(imagePaths)
    
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

def Detectcarowner():
    var=0
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImage\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("carownerdetails\carownerdetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w]) 
            #print(conf)                                  
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                var=0
            else:
                Id='Unknown'                
                tt=str(Id)  
                var=1
            if(conf > 75):
                noOfFile=len(os.listdir("Image Unknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w]) 
                                       
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('s')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Detected\Detected_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    #print(attendance)
    res=attendance
    print(res)
    if var ==1:
       popup()
       var=0       
    else:       
        dd="Face Detected Unlock the door"
        message2.configure(text= res)
        message3.configure(text= dd)


def popup():
    result = messagebox.askquestion("Popup Title", "Do you want to continue?")
    if result == 'yes':
        print("You clicked Yes!")
        dd="Allow to unlock"
        message3.configure(text= dd)
    else:
        print("lock the car")
        dd="Not allow to unlock"
        message3.configure(text= dd)

# clearButton = tk.Button(window, text="Clear", command=clear  ,fg="red"  ,bg="yellow"  ,width=20  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
# clearButton.place(x=950, y=200)
# clearButton2 = tk.Button(window, text="Clear", command=clear2  ,fg="red"  ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
# clearButton2.place(x=950, y=300)    
takeImg = tk.Button(window, text="OwnerImage", command=carownerimage  ,fg="red"   ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
takeImg.place(x=200, y=500)
trainImg = tk.Button(window, text="Train Images", command=TrainImages  ,fg="red"   ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trainImg.place(x=500, y=500)
trackImg = tk.Button(window, text="Detectcarowner", command=Detectcarowner  ,fg="red"   ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="red"   ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=500)
#popup_btn = tk.Button(window, text="Click Me", command=popup)
#popup_btn.pack(pady=20) 
 
window.mainloop()