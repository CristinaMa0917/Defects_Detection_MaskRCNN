from mrcnn.config import Config
from mrcnn.utils import Dataset
import numpy as np

class ChipsConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name

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
    
class ChipsDataset(Dataset):
    def __init__(self,images,masks):
        #new_append 
        self.num_null_mask = 0
        self.all_flaw_size = []
        self.subset = ""
        #original
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
            self.add_image("chips", image_id=i, path=None,chips=chips)
            
    
    def mask_prepare(self,i,return_mask=False): 
        mask_full = self.masks[i,:].reshape((4,224,224))
        bump,dent,dot = mask_full[1,:,:],mask_full[2,:,:],mask_full[3,:,:]   
        flaw_size = [np.sum(bump),np.sum(dent),np.sum(dot)]
        names = ["bump","dent","dot"]

        if not np.sum(flaw_size):
            self.num_null_mask += 1

        if not return_mask:
            chips= []
            for j in range(3):
                if flaw_size[j] >0:
                    chips.append(names[j])
                    self.all_flaw_size.append(flaw_size[j])
            return chips
        else:
            masks = []
            for j in range(3):
                if flaw_size[j] >0:
                    masks.append(mask_full[j+1,:,:])
            if not np.array(masks).shape[0]:
                return np.array(masks)
            else:
                masks = np.array(masks)
                ch = masks.shape[0]
                masks_reshape = np.zeros((224,224,ch))
                for i in range(224):
                    for j in range(224):
                        masks_reshape[i,j,:]=masks[:,i,j]
                return masks_reshape
                      
        
    def load_image(self,image_id):
        image_full = self.images[image_id,:].reshape((6,224,224))
        info = self.image_info[image_id]
        image_sel = np.array([image_full[0,:,:],image_full[1,:,:],image_full[2,:,:]])
        image_reshape = np.zeros((224,224,3))
        for i in range(224):
            for j in range(224):
                image_reshape[i,j,:]=image_sel[:,i,j]
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
        mask = self.mask_prepare(image_id,True)
        
        class_ids = np.array([self.class_names.index(s) for s in chips])
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def info_display(self,count):
        #flaw_size info
        max_flaw_size = np.max(self.all_flaw_size)
        min_flaw_size = np.min(self.all_flaw_size)
        mean_flaw_size = np.mean(self.all_flaw_size)
        
        #flaw_kind_info
        bump_num,dent_num,dot_num = 0,0,0
        for i in range(count):
            chips = self.image_info[i]['chips']
            if "bump" in chips:
                bump_num +=1
            if "dent" in chips:
                dent_num +=1
            if "dot" in chips:
                dot_num += 1    
                
        #display all info
        print("**** info of %s dataset : ***  "%self.subset)
        print("  ratio of null mask images:%d %%"%(self.num_null_mask/count*100))
        print("  max flaw size:",max_flaw_size)
        print("  min flaw size:",min_flaw_size)
        print("  mean flaw size:",mean_flaw_size)
        print("  bump ratio:%d %%"%(bump_num/count*100))
        print("  dent ratio:%d %%"%(dent_num/count*100))
        print("  dot ratio:%d %%"%(dot_num/count*100))
