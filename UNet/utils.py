import h5py
import skimage.transform
from keras.backend import *
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,TensorBoard
import keras.models as KM


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


class ChipsDataset():
    def __init__(self, images, masks):
        self.image_ids = np.arange(len(masks))
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}
        self.images = images
        self.masks = masks
        self.load_chips(len(masks))

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

    def mask_prepare(self, i, return_mask=False):
        mask_full = self.masks[i, :].reshape((4, 224, 224))
        bump, dent, dot = mask_full[1, :, :], mask_full[2, :, :], mask_full[3, :, :]
        flaw_size = [np.sum(bump), np.sum(dent), np.sum(dot)]
        names = ["bump", "dent", "dot"]

        if not return_mask:
            chips = []
            for j in range(3):
                if flaw_size[j] > 0:
                    chips.append(names[j])
            return chips
        else:
            masks = []
            for j in range(3):
                if flaw_size[j] > 0:
                    masks.append(mask_full[j + 1, :, :])
            if not np.array(masks).shape[0]:
                return np.zeros((224,224,1))
            else:
                masks = np.array(masks)
                ch = masks.shape[0]
                masks_reshape = np.zeros((224, 224, ch))
                for i in range(224):
                    for j in range(224):
                        masks_reshape[i, j, :] = masks[:, i, j]
                return masks_reshape

    def load_image(self, image_id):
        image_full = self.images[image_id, :].reshape((6, 224, 224))
        image_sel = np.array([image_full[0, :, :], image_full[1, :, :], image_full[2, :, :]])
        image_reshape = np.zeros((224, 224, 3))
        for i in range(224):
            for j in range(224):
                image_reshape[i, j, :] = image_sel[:, i, j]
        image = skimage.transform.resize(image_reshape, (256, 256), order=1, mode="constant", preserve_range=True)
        return image

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        mask = self.mask_prepare(image_id, True)
        mask = skimage.transform.resize(mask, (256, 256), order=1, mode="constant", preserve_range=True)
        return mask.astype(np.bool)

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

