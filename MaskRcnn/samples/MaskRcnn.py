import os

from MaskRcnn.mrcnn import visualize
from MaskRcnn.mrcnn import utils
import MaskRcnn.mrcnn.model as modellib
import MaskRcnn.samples.coco.coco

coco_Weight = "mask_rcnn_coco.h5"


class InferenceConfig(MaskRcnn.samples.coco.coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class MaskRcnn:
    def __init__(self, ROOT_DIR):
        # Local path to trained weights file
        self.COCO_MODEL_PATH = os.path.join(ROOT_DIR, coco_Weight)
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(self.COCO_MODEL_PATH):
            utils.download_trained_weights(self.COCO_MODEL_PATH)

        # Directory of images to run detection on
        IMAGE_DIR = os.path.join(ROOT_DIR, "images")

        self.config = InferenceConfig()
        self.config.display()
        self.MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        self.load_model()

    def load_model(self):
        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=self.MODEL_DIR, config=self.config)

        # Load weights trained on MS-COCO
        self.model.load_weights(self.COCO_MODEL_PATH, by_name=True)

        ############

        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light',
                       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                       'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                       'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                       'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                       'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush']

    def get_prediction(self, image):

        # Run detection
        results = self.model.detect([image], verbose=1)
        return results

    def get_predictionDrawed(self, frame, detection):
        return visualize.get_display_instances_image(frame, detection['rois'], detection['masks'], detection['class_ids'],
                                    self.class_names, detection['scores'])






