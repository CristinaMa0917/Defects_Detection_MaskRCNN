from mrcnn_6.utils import Dataset
import numpy as np
class ChipsConfig(object):

    NAME = "chips"

    # CH
    BACKBONE = "resnet50"
    MINI_MASK_SHAPE = (28, 28)
    DETECTION_MAX_INSTANCES = 10
    ROI_POSITIVE_RATIO = 0.1
    USE_MINI_MASK = True

    # TRY CH
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1  # CH
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 20

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4, 8, 16)  # anchor side in pixels # CH

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 40  # CH


    # Only useful if you supply a callable to BACKBONE. Should compute
    # the shape of each layer of the FPN Pyramid.
    # See model.compute_backbone_shapes
    COMPUTE_BACKBONE_SHAPE = None

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 256

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 200
    POST_NMS_ROIS_INFERENCE = 100

    IMAGE_RESIZE_MODE = "square"

    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # Howver, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 0

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.6

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.001

    LEARNING_MOMENTUM = 0.9
    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 2.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 2.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch sizeF
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, 3])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    # def display(self):
    #     """Display Configuration values."""
    #     print("\nConfigurations:")
    #     for a in dir(self):
    #         if not a.startswith("__") and not callable(getattr(self, a)):
    #             print("{:30} {}".format(a, getattr(self, a)))
    #     print("\n")


class ChipsDataset(Dataset):
    def __init__(self, images=0,masks=0,class_map=None):
        # new_append
        self.num_null_mask = 0
        self.all_flaw_size = []
        # original
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}
        self.images = images
        self.masks = masks

    def load_chips(self, count):
        # Add classes
        self.add_class("chips", 1, "bump")
        self.add_class("chips", 2, "dent")
        self.add_class("chips", 3, "dot")

        # Add images
        # Images are generated on the fly in load_image().
        for i in range(count):
            chips = self.mask_prepare(i)
            self.add_image("chips", image_id=i, path=None, chips=chips)

    def mask_prepare(self,i, return_mask=False):
        mask_full = self.masks[i,:].reshape(4,224,224)
        bump, dent, dot = mask_full[1, :, :], mask_full[2, :, :], mask_full[3, :, :]
        flaw_size = [np.sum(bump), np.sum(dent), np.sum(dot)]
        names = ["bump", "dent", "dot"]

        if not np.sum(flaw_size):
            self.num_null_mask += 1
            # print("==== no flaw for image %d ===="%i)

        if not return_mask:
            chips = []
            for j in range(3):
                if flaw_size[j] > 0:
                    chips.append(names[j])
                    self.all_flaw_size.append(flaw_size[j])
            return chips
        else:
            masks = []
            for j in range(3):
                if flaw_size[j] > 0:
                    masks.append(mask_full[j + 1, :, :])
            if not np.array(masks).shape[0]:
                return np.array(masks)
            else:
                masks = np.array(masks)
                masks_reshape = np.zeros((224,224,masks.shape[0]))
                for i in range(224):
                    for j in range(224):
                        masks_reshape[i,j,:]=masks[:,i,j]
                return masks_reshape

    def load_image(self, image_id):
        image_full = self.images[image_id,:].reshape((6,224,224))
        image_reshape=np.zeros((224,224,6))
        for i in range(6):
            image_reshape[:,:,i]=image_full[i,:,:]
        return image_reshape

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "chips":
            return info["chips"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        chips = info['chips']
        mask = self.mask_prepare(image_id, True)
        class_ids = np.array([self.class_names.index(s) for s in chips])
        return mask.astype(np.bool), class_ids.astype(np.int32)
