import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import threading
import matplotlib.pyplot as plt
from PIL import ImageSequence, ImageTk
from matplotlib import cm as c
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


width = 720
height = 480
loading = True

class detectionWindow:
    def __init__(self, window, window_title, image = None):
        self.window = tkinter.Toplevel(window)
        self.window.title(window_title)
        self.window.resizable(False, False)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(self.window, width=width, height=height)
        self.canvas.pack()

        src = cv2.imread('background/loading2.png', cv2.IMREAD_UNCHANGED)
        frame_resized = cv2.resize(src, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_resized))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)



    def setImage(self, image):

        frame_resized = cv2.resize(image, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_resized))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

"""
    def showPlot(self, ans):
        a = plt.imshow(ans.reshape(ans.shape[1], ans.shape[2]), cmap=c.jet)
        a.get_array()
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(a.get_array()))

        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

    def matplotCanvas(self, ans):
        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)
        a.plot([1,2,3,4,5,6,7,8] , [5,6,1,3,8,9,3,5])

        self.canvas1 = FigureCanvasTkAgg(f, self.window)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(side = tkinter.BOTTOM, fill=tkinter.BOTH, expand=True)
"""