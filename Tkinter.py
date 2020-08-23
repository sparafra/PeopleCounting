import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import threading

from MaskRcnn.samples.MaskRcnn import MaskRcnn
from Yolo.Yolo_V4 import Yolo_V4

#AlgorithmIndex = 1
width = 1280
height = 720

peopleCount = "0"
fps = "FPS: 0"

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        self.maskRcnn = MaskRcnn("MaskRcnn/")
        self.yolo = Yolo_V4("Yolo/")

        self.threadAI = None
        self.AlgorithmIndex = 0


        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

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


        self.lblCount = tkinter.Label(window, text="0", bg = "dark green", fg="white", height=5)
        self.lblCount.pack(side=tkinter.LEFT, expand=True)

        self.lblFPS = tkinter.Label(window, text="FPS: 0", bg="yellow", fg="red", height=5)
        self.lblFPS.pack(side=tkinter.LEFT, expand=True)
        #lblCount = tkinter.Label(window, text=peopleCount, bg="dark green", fg="white", height=5)
        #lblCount.pack(side=tkinter.LEFT, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def change_algorithm(self):
        self.AlgorithmIndex = self.v.get()
        print("Change Alg: " + str(self.v.get()))



    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            if not self.threadAI or not self.threadAI.is_alive():
                if self.AlgorithmIndex == 1:
                    self.threadAI = ThreadAI("Thread1", frame, self.maskRcnn, self.AlgorithmIndex, self.lblCount, self.lblFPS)
                    self.threadAI.start()
                    
                elif self.AlgorithmIndex == 2:
                    self.threadAI = ThreadAI("Thread2", frame, self.yolo, self.AlgorithmIndex, self.lblCount, self.lblFPS)
                    self.threadAI.start()
                elif self.AlgorithmIndex == 3:
                    #CSRNET
                    print("CSRNET")


            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        #self.lblCount['text'] = peopleCount
        #print(peopleCount)
        #print(self.v.get())
        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                resize = cv2.resize(frame, (width, height))
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(resize, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (None, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

class ThreadAI(threading.Thread):
    def __init__(self, nome, image, algorithm, AlgorithmIndex, labelPeople, labelFPS):
        threading.Thread.__init__(self)
        self.nome = nome
        self.image = image
        self.algorithm = algorithm
        self.labelPeople = labelPeople
        self.labelFPS = labelFPS
        self.AlgorithmIndex = AlgorithmIndex
    def run(self):

        self.prev_time = time.time()

        # Run detection
        results = self.algorithm.get_prediction(self.image)
        self.fps = 1 / (time.time() - self.prev_time)
        self.labelFPS['text'] = "Time: {}".format(self.fps)

        self.peopleCount = "0"
        #print("AlgIndex: " + str(self.AlgorithmIndex))
        if self.AlgorithmIndex == 1:
            # Visualize results
            r = results[0]

            print("Thread '" + self.name + "' " + str(r['class_ids'].size))
            self.peopleCount = str(r['class_ids'].size)
        elif self.AlgorithmIndex == 2:
            #print("YOLO")
            nPerson = 0
            for label, confidence, bbox in results:
                if label == "person":
                    nPerson += 1
            print("Thread '" + self.nome + "' " + str(nPerson))
            self.peopleCount = str(nPerson)
        elif self.AlgorithmIndex == 3:
            self.peopleCount = "0"

        self.labelPeople['text'] = self.peopleCount

        #print(peopleCount)

        # Rilascio del lock
        #threadLock.release()


# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV", "Video/Test1.mp4")