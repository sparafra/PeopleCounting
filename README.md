# PeopleCounting

Importare il file "yolov4.weights" nella cartella "Yolo". https://drive.google.com/file/d/1HbA1DOfc7GK5-tQZasjUnzLch8EfIkIP/view?usp=sharing
Importare il file “mask_rcnn_coco.h5” nella cartella “MaskRcnn”. https://drive.google.com/file/d/1QeQxye1AHmXTbIJybcw34VDYRH_dzpp9/view?usp=sharing

Yolo V4: https://github.com/AlexeyAB/darknet

Se vengono generati errori si consiglia di clonare il progetto di Yolo V4 e compilare i sorgenti tramite il file “Makefile” che è possibile eseguire da linux tramite il comando “make” lanciato dalla directory in cui è situato il file “Makefile”.

Per usufruire dell’elaborazione GPU e Cuda bisogna modificare il file “Makefile” e cambiare le seguenti righe:
  GPU=1
  CUDNN=1
  CUDNN_HALF=1
  OPENCV=1

In base al tipo di GPU che si ha bisogna scegliere la riga da rendere visibile rimuovendo il simbolo di commento “#”.

Nel mio caso avendo una GTX 1050 devo rendere visibile la seguente riga (41):
  ARCH= -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61

Successivamente bisogna eseguire il file “build.sh”, questo può essere fatto eseguendo nel terminale il comando “./build.sh”
