import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import threading

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

        self.video_source = video_source

        self.showPredictionAnalysis = None
        self.detectionWindow = None

        self.maskRcnn = MaskRcnn("MaskRcnn/")
        self.yolo = Yolo_V4("Yolo/")
        self.csrNet = CSRNet("CSRNet/")
        self.threadAI = None
        self.AlgorithmIndex = 0

        self.showPredictionAnalysis = False




        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        self.vid.start()

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = width, height = height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        #self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        #self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        self.v = tkinter.IntVar()
        self.v.set(1)
        self.change_algorithm()

        self.btn_MaskRcnn = tkinter.Radiobutton(window, text="MaskRcnn", variable=self.v, value=1,
                                                command=self.change_algorithm)
        self.btn_MaskRcnn.pack(side=tkinter.LEFT, expand=True)

        self.btn_Yolo = tkinter.Radiobutton(window, text="YOLO V4", variable=self.v, value=2,
                                            command=self.change_algorithm)
        self.btn_Yolo.pack(side=tkinter.LEFT, expand=True)

        self.btn_CSRNet = tkinter.Radiobutton(window, text="CSRNet", variable=self.v, value=3,
                                              command=self.change_algorithm)
        self.btn_CSRNet.pack(side=tkinter.LEFT, expand=True)

        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Prediction Render", width=50, command=self.showAnalysis)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        self.lblCount = tkinter.Label(window, text="0", bg = "dark green", fg="white", height=5)
        self.lblCount.pack(side=tkinter.LEFT, expand=True)

        self.lblFPS = tkinter.Label(window, text="FPS: 0", bg="yellow", fg="red", height=5)
        self.lblFPS.pack(side=tkinter.LEFT, expand=True)
        #lblCount = tkinter.Label(window, text=peopleCount, bg="dark green", fg="white", height=5)
        #lblCount.pack(side=tkinter.LEFT, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay =1 #15
        self.update()

        self.window.mainloop()

    def showAnalysis(self):
        self.showPredictionAnalysis = not self.showPredictionAnalysis
        if self.threadAI:
            self.threadAI.showAnalysis(self.showPredictionAnalysis)
        print(self.showPredictionAnalysis)

    def createNewWindow(self):
        self.detectionWindow = tkinter.Toplevel(self.window)
        self.detectionWindow.title("Detection Result")

        # Create a canvas that can fit the above video source size
        self.canvasDetected = tkinter.Canvas(self.detectionWindow, width=width, height=height)
        self.canvasDetected.pack()


    def setImageDetected(self, image):
        self.photoDetected = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
        self.canvasDetected.create_image(0, 0, image=self.photoDetected, anchor=tkinter.NW)

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def change_algorithm(self):
        self.AlgorithmIndex = self.v.get()
        print("Change Alg: " + str(self.v.get()))
        if self.threadAI:
            self.threadAI.stop()



    def update(self):
        start_time = time.time()  # start time of the loop

        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            if not self.threadAI or not self.threadAI.is_alive():
                if self.AlgorithmIndex == 1:
                    self.threadAI = ThreadAI("Thread1", self.vid, frame, self.maskRcnn, self.AlgorithmIndex, self.lblCount, self.lblFPS, self.detectionWindow, self.showPredictionAnalysis)
                    self.threadAI.start()
                elif self.AlgorithmIndex == 2:
                    self.threadAI = ThreadAI("Thread2", self.vid, frame, self.yolo, self.AlgorithmIndex, self.lblCount, self.lblFPS, self.detectionWindow, self.showPredictionAnalysis)
                    self.threadAI.start()
                elif self.AlgorithmIndex == 3:
                    self.threadAI = ThreadAI("Thread3", self.vid, frame, self.csrNet, self.AlgorithmIndex, self.lblCount,
                                             self.lblFPS, self.detectionWindow, self.showPredictionAnalysis)
                    self.threadAI.start()



            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            #print("FPS: ", 1.0 / (time.time() - start_time))  # FPS = 1 / time to process loop

        if self.showPredictionAnalysis and not self.detectionWindow:
            self.detectionWindow = detectionWindow(self.window, "Detection")

        self.window.after(self.delay, self.update)

class MyVideoCapture(threading.Thread):
    def __init__(self, video_source=0):
        threading.Thread.__init__(self)
        # Open the video source
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

    def run(self):
        while self.vid.isOpened():
            start_time = time.time()  # start time of the loop
            timeout = 0
            self.ret, self.frame = self.vid.read()
            #time.sleep(0.01)
            end_time = time.time()
            if (end_time - start_time) < (1/self.fps_info):
                timeout = (1/self.fps_info) - (time.time() - start_time);
            #print(timeout)
            time.sleep(timeout)
            #print("time to expire: ", timeout)
            #print("Time elapsed: ", (time.time() - start_time))
            print("FPS: ", 1.0 / (time.time() - start_time))  # FPS = 1 / time to process loop


    def get_frame(self):
        if self.vid.isOpened():
            if self.ret:
                resize = cv2.resize(self.frame, (width, height))

                return (self.ret, cv2.cvtColor(resize, cv2.COLOR_BGR2RGB))
            else:
                return (self.ret, None)
        else:
            return (None, None)


    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

class ThreadAI(threading.Thread):
    def __init__(self, nome, vid, image, algorithm, AlgorithmIndex, labelPeople, labelFPS, detectionWindow, showPredictionAnalysis):
        threading.Thread.__init__(self)
        self.nome = nome
        self.image = image
        self.algorithm = algorithm
        self.labelPeople = labelPeople
        self.labelFPS = labelFPS
        self.AlgorithmIndex = AlgorithmIndex
        self.stopVar = False
        self.vid = vid
        self.detectionWindow = detectionWindow
        self.showPredictionAnalysis = showPredictionAnalysis

    def run(self):

        while not self.stopVar:
            ret, frame = self.vid.get_frame()

            self.prev_time = time.time()

            # Run detection
            self.results = self.algorithm.get_prediction(frame)

            self.fps = 1 / (time.time() - self.prev_time)
            self.labelFPS['text'] = "FPS: {}".format(self.fps)

            self.peopleCount = "0"

            if self.AlgorithmIndex == 1:
                # Visualize results
                r = self.results[0]

                #print("Thread '" + self.name + "' " + str(r['class_ids'].size))
                print(self.showPredictionAnalysis)
                if self.showPredictionAnalysis:
                    self.detectionWindow.setImage(self.algorithm.get_predictionDrawed(frame, r))

                self.peopleCount = str(r['class_ids'].size)
            elif self.AlgorithmIndex == 2:
                nPerson = 0
                for label, confidence, bbox in self.results:
                    if label == "person":
                        nPerson += 1
                if self.showPredictionAnalysis:
                    self.detectionWindow.setImage(self.algorithm.get_predictionDrawed(frame, self.results))

                #print("Thread '" + self.name + "' " + str(nPerson))
                self.peopleCount = str(nPerson)
            elif self.AlgorithmIndex == 3:
                #print("Thread '" + self.name + "' " + str(self.results))
                #self.detectionWindow.showPlot(self.algorithm.get_predictionDrawed(frame))
                self.peopleCount = str(self.results)

            self.labelPeople['text'] = "People: " + self.peopleCount

    def stop(self):
        self.stopVar = True
    def showAnalysis(self, bool):
        self.showPredictionAnalysis = bool




# Create a window and pass it to the Application object
App(tkinter.Tk(), "People Counting", "Video/Test3.mp4")