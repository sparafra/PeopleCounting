import tkinter
from tkinter import filedialog

import cv2
import PIL.Image, PIL.ImageTk
import time
import threading
from collections import deque
import matplotlib
import numpy as np
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import cm as c
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from CSRNet.CSRNet import CSRNet
from MaskRcnn.samples.MaskRcnn import MaskRcnn
from Yolo.Yolo_V4 import Yolo_V4
from detectionWindow import detectionWindow

width = 1280
height = 720

peopleCount = "0"
fps = "FPS: 0"

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        #self.window.wait_visibility(self.window)
        #self.window.wm_attributes('-alpha', 0.3)
        #self.window.configure(background='grey')

        #Image Background
        src = cv2.imread('background/background.png', cv2.IMREAD_UNCHANGED)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(src, (1920, 1080),
                                   interpolation=cv2.INTER_LINEAR)
        background_image = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_resized))
        background_label = tkinter.Label(window, image=background_image)
        background_label.place(x=0, y=0, relwidth=1, relheight=1)


        self.video_source = video_source

        self.showPredictionAnalysis = None
        self.detectionWindow = None

        #Define the Directory of algorithms
        self.yolo = Yolo_V4("Yolo/")
        self.maskRcnn = MaskRcnn("MaskRcnn/")
        self.csrNet = CSRNet("CSRNet/")
        self.threadAI = None
        self.AlgorithmIndex = 0

        #Prediction Window
        self.showPredictionAnalysis = False

        self.file_path = filedialog.askopenfilename(filetypes = (("MP4 Files","*.mp4"),))

        # open video source (MyVideoCapture is on new thread)
        self.vid = MyVideoCapture(self.file_path)

        self.vid.start()

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = 850, height = 600)
        #self.canvas.pack(side=tkinter.LEFT, expand=True)
        self.canvas.grid(row = 0, column = 0, columnspan=3, rowspan = 3, sticky = tkinter.W, padx = 10, pady = 2)

        self.canvas_prediction = tkinter.Canvas(window, width=850, height=600)
        #self.canvas_prediction.pack()
        #self.canvas_prediction.pack(side=tkinter.RIGHT, expand=True)
        self.canvas_prediction.grid(row = 0, column = 5,columnspan=3, rowspan = 3, sticky = tkinter.E, pady = 2)


        src = cv2.imread('background/loading2.png', cv2.IMREAD_UNCHANGED)
        frame_resized = cv2.resize(src, (850, 600),
                                   interpolation=cv2.INTER_LINEAR)
        photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_resized))
        self.canvas_prediction.create_image(0, 0, image=photo, anchor=tkinter.NW)

        #V is the variable for the three radiobutton
        self.v = tkinter.IntVar()
        self.v.set(1)
        #update the algorithm index
        self.change_algorithm()

        self.btn_MaskRcnn = tkinter.Radiobutton(window, text="MaskRcnn", variable=self.v, value=1,
                                                command=self.change_algorithm)
        #self.btn_MaskRcnn.pack(side=tkinter.LEFT, expand=True)
        self.btn_MaskRcnn.grid(row = 6, column = 5, sticky = tkinter.W, pady = 10)

        self.btn_Yolo = tkinter.Radiobutton(window, text="YOLO V4", variable=self.v, value=2,
                                            command=self.change_algorithm)
        #self.btn_Yolo.pack(side=tkinter.LEFT, expand=True)
        self.btn_Yolo.grid(row = 6, column = 6, sticky = tkinter.W, pady = 10)

        self.btn_CSRNet = tkinter.Radiobutton(window, text="CSRNet", variable=self.v, value=3,
                                              command=self.change_algorithm)
        #self.btn_CSRNet.pack(side=tkinter.LEFT, expand=True)
        self.btn_CSRNet.grid(row = 6, column = 7, sticky = tkinter.W, pady = 10)

        self.lblCount = tkinter.Label(window, text="People: ...", bg = "dark green", fg="white", height=5, width=15)
        #self.lblCount.pack(side=tkinter.LEFT, expand=True)
        self.lblCount.grid(row = 0, column = 4, sticky = tkinter.W, pady = 2, padx = 5)

        self.lblFPS = tkinter.Label(window, text="FPS: 0", bg="orange", fg="white", height=5, width=15)
        #self.lblFPS.pack(side=tkinter.LEFT, expand=True)
        self.lblFPS.grid(row = 2, column = 4, sticky = tkinter.W, pady = 2, padx = 5)

        self.lblFrame = tkinter.Label(window, text="FRAME: 0", bg="yellow", fg="red", height=5)
        # self.lblFPS.pack(side=tkinter.LEFT, expand=True)
        self.lblFrame.grid(row=6, column=1, sticky=tkinter.E, pady=2, padx=5)

        self.lblFrame_detected = tkinter.Label(window, text="Examinated: ...", bg="yellow", fg="red", height=5, width=15)
        # self.lblFPS.pack(side=tkinter.LEFT, expand=True)
        self.lblFrame_detected.grid(row=1, column=4, sticky=tkinter.W, pady=2, padx=5)

        #str(self.file_path).split("/")[len(str(self.file_path).split("/")) - 1]
        #Get the file name

        self.lblPath = tkinter.Label(window, text=str(self.file_path).split("/")[len(str(self.file_path).split("/"))-1],  fg="black", height=3)
        # self.lblFPS.pack(side=tkinter.LEFT, expand=True)
        self.lblPath.grid(row=6, column=2, sticky=tkinter.W, pady=2, padx=115)

        pause_src = cv2.imread('background/Icon/pause.png', cv2.IMREAD_UNCHANGED)
        pause_resized = cv2.resize(pause_src, (50, 50),
                                   interpolation=cv2.INTER_LINEAR)
        pause = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(pause_resized))
        resume_src = cv2.imread('background/Icon/resume.png', cv2.IMREAD_UNCHANGED)
        resume_resized = cv2.resize(resume_src, (50, 50),
                                   interpolation=cv2.INTER_LINEAR)
        resume = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(resume_resized))
        stop_src = cv2.imread('background/Icon/stop.png', cv2.IMREAD_UNCHANGED)
        stop_resized = cv2.resize(stop_src, (50, 50),
                                    interpolation=cv2.INTER_LINEAR)
        stop = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(stop_resized))

        file_src = cv2.imread('background/Icon/folder.png', cv2.IMREAD_UNCHANGED)
        file_resized = cv2.resize(file_src, (50, 50),
                                  interpolation=cv2.INTER_LINEAR)
        file = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(file_resized))

        self.btn_pause = tkinter.Button(window, text="PAUSE", image = pause, height = 50, width = 50,
                                                    command=self.pauseVideo)

        self.btn_pause.grid(row = 6, column = 0, sticky = tkinter.E, pady = 2, padx = 0)

        self.btn_resume = tkinter.Button(window, text="RESUME", image = resume, height = 50, width = 50,
                                        command=self.resumeVideo)
        self.btn_resume.grid(row=6, column=0, sticky=tkinter.E, pady = 2, padx=50)

        self.btn_stop = tkinter.Button(window, text="STOP", image = stop, height = 50, width = 50,
                                        command=self.stopVideo)
        self.btn_stop.grid(row=6, column=0, sticky=tkinter.E, pady = 2, padx=100)

        self.btn_file = tkinter.Button(window, text="", image = file, height=50, width=50,
                                       command=self.fileDialog)
        self.btn_file.grid(row=6, column=2, sticky=tkinter.W, pady=2, padx=55)

        #self.btn_renderPrediction = tkinter.Button(window, text="Prediction Render", bg="orange", width=50,
                                                   #command=self.showAnalysis)
        #self.btn_renderPrediction.pack(anchor=tkinter.CENTER, expand=True)
        #self.btn_renderPrediction.grid(row = 6, column = 4, sticky = tkinter.W, pady = 2)


        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay =1 #15
        self.update()

        self.window.mainloop()

    def fileDialog(self):
        self.file_path = filedialog.askopenfilename(filetypes = (("MP4 Files","*.mp4"),))
        self.lblPath['text'] = str(self.file_path).split("/")[len(str(self.file_path).split("/"))-1]
        #print(str(self.file_path).split("/"))
        self.stopVideo()
        self.vid.startVideo(self.file_path)
        self.btn_resume['state'] = "disabled"
        self.btn_pause['state'] = "normal"
        self.btn_stop['state'] = "normal"


    def pauseVideo(self):
        self.vid.pause_capture()
        self.btn_pause['state'] = "disabled"
        self.btn_resume['state'] = "normal"
        self.btn_stop['state'] = "normal"

    def resumeVideo(self):
        self.vid.resume_capture()
        self.btn_resume['state'] = "disabled"
        self.btn_pause['state'] = "normal"
        self.btn_stop['state'] = "normal"


    def stopVideo(self):
        self.vid.stop_capture()
        self.btn_pause['state'] = "disabled"
        self.btn_stop['state'] = "disabled"
        self.btn_resume['state'] = "normal"

    def showAnalysis(self):
        #Set showAnalysis boolean and create a new window for the detection result. (the window will be passed to the AI thread)
        self.showPredictionAnalysis = not self.showPredictionAnalysis
        if self.showPredictionAnalysis and not self.detectionWindow:
            self.detectionWindow = detectionWindow(self.window, "Detection")
            self.threadAI.setDetectionWindow(self.detectionWindow)
        if self.threadAI:
            self.threadAI.showAnalysis(self.showPredictionAnalysis)




    def change_algorithm(self):
        #Stop the AI thread who running with others algorithm
        self.AlgorithmIndex = self.v.get()
        print("Change Alg: " + str(self.v.get()))
        if self.threadAI:
            self.threadAI.stop()



    def update(self):
        #start_time = time.time()  # start time of the loop

        # Get a frame from the video source
        ret, frame, number = self.vid.get_frame()
        if self.vid.isStop():
            self.btn_pause['state'] = "disabled"
            self.btn_stop['state'] = "disabled"
            self.btn_resume['state'] = "normal"


        if ret:
            if not self.threadAI or not self.threadAI.is_alive():
                if self.AlgorithmIndex == 1:
                    self.threadAI = ThreadAI("Thread1", self.vid, frame, self.maskRcnn, self.AlgorithmIndex, self.lblCount, self.lblFPS, self.lblFrame_detected, self.detectionWindow, self.showPredictionAnalysis, self.canvas_prediction)
                    self.threadAI.start()
                elif self.AlgorithmIndex == 2:
                    self.threadAI = ThreadAI("Thread2", self.vid, frame, self.yolo, self.AlgorithmIndex, self.lblCount, self.lblFPS, self.lblFrame_detected, self.detectionWindow, self.showPredictionAnalysis, self.canvas_prediction)
                    self.threadAI.start()
                elif self.AlgorithmIndex == 3:
                    self.threadAI = ThreadAI("Thread3", self.vid, frame, self.csrNet, self.AlgorithmIndex, self.lblCount,
                                             self.lblFPS, self.lblFrame_detected, self.detectionWindow, self.showPredictionAnalysis, self.canvas_prediction)
                    self.threadAI.start()


            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            self.lblFrame['text'] = "FRAME: " + str(number)
            #print("FRAME: ", number)

            #print("FPS: ", 1.0 / (time.time() - start_time))  # FPS = 1 / time to process loop


        self.window.after(self.delay, self.update)

class MyVideoCapture(threading.Thread):
    def __init__(self, video_source=0):
        threading.Thread.__init__(self)
        # Open the video source
        self.video_source = video_source
        self.initialize(self.video_source)

    def initialize(self, video_source):
        self.vid = cv2.VideoCapture(video_source)

        self.fps_info = self.vid.get(cv2.CAP_PROP_FPS)
        print(self.fps_info)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.ret = None
        self.frame = None
        self.nFrame = 0

        self.pauseVideo = False
        self.stopVideo = None

    def run(self):
        while self.vid.isOpened():
            if not self.pauseVideo and not self.stopVideo:
                start_time = time.time()  # start time of the loop
                timeout = 0

                self.ret, self.frame = self.vid.read()
                self.nFrame = self.vid.get(1)

                end_time = time.time()

                if (end_time - start_time) < (1/self.fps_info):
                    timeout = (1/self.fps_info) - (time.time() - start_time);

                time.sleep(timeout)

                print("FPS: ", 1.0 / (time.time() - start_time))  # FPS = 1 / time to process loop
                #print("FRAME THREAD: ", self.nFrame)
                if self.nFrame == self.vid.get(7):
                    self.stop_capture()

    def get_frame(self):
        if self.vid.isOpened() and not self.pauseVideo and not self.stopVideo:
            if self.ret:
                resize = cv2.resize(self.frame, (width, height))
                return (self.ret, cv2.cvtColor(resize, cv2.COLOR_BGR2RGB), self.nFrame)
            else:
                return (self.ret, None, None)
        else:
            return (None, None, None)

    def pause_capture(self):
        self.pauseVideo = True

    def resume_capture(self):
        if self.pauseVideo:
            self.pauseVideo = False
        else:
            if self.stopVideo:
                self.stopVideo = False
            print("START")
    def stop_capture(self):
        self.stopVideo = True
        self.vid.set(1, 0)

    def startVideo(self, path):
        self.initialize(path)

    def isStop(self):
        return self.stopVideo

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

class ThreadAI(threading.Thread):
    def __init__(self, nome, vid, image, algorithm, AlgorithmIndex, labelPeople, labelFPS, labelFrame, detectionWindow, showPredictionAnalysis, canvas_prediction):
        threading.Thread.__init__(self)

        self.nome = nome
        self.image = image
        self.algorithm = algorithm
        self.labelPeople = labelPeople
        self.labelFPS = labelFPS
        self.labelFrame = labelFrame
        self.AlgorithmIndex = AlgorithmIndex
        self.stopVar = False
        self.vid = vid
        self.detectionWindow = detectionWindow
        self.showPredictionAnalysis = showPredictionAnalysis
        self.canvas_prediction = canvas_prediction


    def run(self):

        while not self.stopVar:
            ret, frame, number = self.vid.get_frame()

            if ret:
                self.labelFrame['text'] = "Examinated: " + str(number)


                self.prev_time = time.time()

                # Run detection
                self.results = self.algorithm.get_prediction(frame)

                #Show fps for the detection
                self.fps = 1 / (time.time() - self.prev_time)
                self.labelFPS['text'] = "FPS: {}".format(round(self.fps, 2))

                self.peopleCount = "0"

                if self.AlgorithmIndex == 1:

                    # Visualize results
                    r = self.results[0]

                    frame_prediction = self.algorithm.get_predictionDrawed(frame, r)
                    frame_resized = cv2.resize(frame_prediction, (width, height),
                                               interpolation=cv2.INTER_LINEAR)
                    self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_resized))
                    self.canvas_prediction.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

                    #self.savePrediction(frame, frame_prediction, str(number))
                    #if self.showPredictionAnalysis:
                    #    self.detectionWindow.setImage(self.algorithm.get_predictionDrawed(frame, r))

                    self.peopleCount = str(r['class_ids'].size)
                elif self.AlgorithmIndex == 2:
                    nPerson = 0
                    for label, confidence, bbox in self.results:
                        if label == "person":
                            nPerson += 1

                    frame_prediction = self.algorithm.get_predictionDrawed(frame, self.results)
                    frame_resized = cv2.resize(frame_prediction, (width, height),
                                               interpolation=cv2.INTER_LINEAR)
                    self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_resized))
                    self.canvas_prediction.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
                    #self.savePrediction(frame, frame_prediction, str(number))

                    #if self.showPredictionAnalysis:
                    #    self.detectionWindow.setImage(self.algorithm.get_predictionDrawed(frame, self.results))

                    self.peopleCount = str(nPerson)
                elif self.AlgorithmIndex == 3:
                    count, img, hmap = self.results

                    self.peopleCount = str(count)

                    #Show the heatmap preview with plot
                    src = cv2.imread('testP.png', cv2.IMREAD_UNCHANGED)
                    frame_resized = cv2.resize(src, (850, 600),
                                               interpolation=cv2.INTER_LINEAR)
                    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_resized))
                    self.canvas_prediction.create_image(0, 0, image=photo, anchor=tkinter.NW)

                self.labelPeople['text'] = "People: " + self.peopleCount

    def stop(self):
        self.stopVar = True
    def showAnalysis(self, bool):
        self.showPredictionAnalysis = bool
    def setDetectionWindow(self, detectionWindow):
        self.detectionWindow = detectionWindow

    def getPrediction(self):
        return self.results

    def savePrediction(self, frame, framePredicted, name):
        imageOrig = PIL.Image.fromarray(frame)
        imageOrig.save(name+".png")
        imagePred = PIL.Image.fromarray(framePredicted)
        imagePred.save(name + "P.png")


# Create a window and pass it to the Application object
App(tkinter.Tk(), "People Counting", "Video/Test3.mp4")