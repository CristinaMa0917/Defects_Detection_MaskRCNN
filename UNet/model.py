import h5py
from keras.backend import *
import datetime
from keras.layers import *
from keras.optimizers import Adam
import re
from keras.callbacks import ModelCheckpoint, TensorBoard
import keras.models as KM


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

def unet(input_size=(256, 256, 3)):# 1024/8 = 128
    inputs = Input(input_size)
    conv1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = Concatenate(axis=3)([drop4, up6])

    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = KM.Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    return model



def identity_block(input,filters):
    conv1 = Conv2D(filters[0],kernel_size=(1,1),padding='same')(input)
    conv1 = BatchNormalization()(conv1,training=True)
    conv1 = Activation('relu')(conv1)

    conv2 = Conv2D(filters[1],kernel_size=(3,3),padding='same')(conv1)
    conv2 = BatchNormalization()(conv2,training=True)
    conv2 = Activation('relu')(conv2)

    conv3 = Conv2D(filters[2],kernel_size=(1,1),padding='same')(conv2)
    conv3 = BatchNormalization()(conv3,training=True)

    x = Add()([conv3,input])
    x = Activation('relu')(x)
    return x

def conv_block(input,filters):
    conv1 = Conv2D(filters[0],kernel_size=(1,1),strides=(2,2))(input)
    conv1 = BatchNormalization()(conv1,training=True)
    conv1 = Activation('relu')(conv1)

    conv2 = Conv2D(filters[1],kernel_size=(3,3),padding='same')(conv1)
    conv2 = BatchNormalization()(conv2,training = True)
    conv2 = Activation('relu')(conv2)

    conv3 = Conv2D(filters[2],kernel_size=(1,1),padding='same')(conv2)
    conv3 = BatchNormalization()(conv3,training=True)

    input_sub = Conv2D(filters[2],kernel_size=(1,1),strides=(2,2))(input)
    input_sub = BatchNormalization()(input_sub,training=True)

    x = Add()([conv3,input_sub])
    x = Activation('relu')(x)
    return x

def unet_res(input_size=(256,256,3)):  # 1024/8 reduced to 8 channels
    inputs = Input(input_size)
    x = Conv2D(8,(3,3),strides=(1,1),padding='same')(inputs)
    x = BatchNormalization()(x,training=True)
    c1 = Activation('relu')(x) #[256,256,64]
    pool1 = MaxPool2D(pool_size=(2,2))(c1)

    c2 = conv_block(c1,[8,8,16])
    c2 = identity_block(c2,[8,8,16])
    c2 = identity_block(c2,[8,8,16]) #[128,,128]

    c3 = conv_block(c2,[16,16,32])
    c3 = identity_block(c3,[16,16,32])
    c3 = identity_block(c3,[16,16,32])
    c3 = identity_block(c3,[16,16,32])#[64,,256]

    c4 = conv_block(c3,[32,32,64])
    c4 = identity_block(c4,[32,32,64])
    c4 = identity_block(c4,[32,32,64])
    c4 = identity_block(c4,[32,32,64])
    c4 = identity_block(c4,[32,32,64])
    c4 = identity_block(c4,[32,32,64]) #[32,,512]

    c5 = conv_block(c4,[64,64,128])
    c5 = identity_block(c5,[64,64,128])
    c5 = identity_block(c5,[64,64,128]) #[16,,1024]

    up6 = UpSampling2D((2,2))(c5)#[32,512]
    up6 = Conv2D(64,(2,2),padding='same',activation='relu')(up6)#[32,512]
    merge6 = Concatenate(axis=3)([up6,c4])#[32,1024]

    up7 = UpSampling2D((2,2))(merge6)
    up7 = Conv2D(32,(2,2),padding='same',activation='relu')(up7)
    merge7 = Concatenate(axis=3)([up7,c3])#[64,512]

    up8 = UpSampling2D((2,2))(merge7)
    up8 = Conv2D(16,2,padding='same',activation='relu')(up8)
    merge8 = Concatenate(axis=3)([up8,c2]) #[128,128*2]

    up9 = UpSampling2D((2,2))(merge8)
    up9 = Conv2D(8,2,padding='same',activation='relu')(up9)
    merge9 = Concatenate(axis=3)([up9,c1]) #[256,64*2}

    conv9 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = KM.Model(inputs=inputs,outputs=conv10)
    model.compile(optimizer=Adam(lr=1e-4),loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    return model

class UNET():
    def __init__(self,mode,model_dir):
        self.type = mode
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build()

    def set_log_dir(self):
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if self.model_dir:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/[\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, self.model_dir)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        self.log_dir = self.model_dir

        # Create log_dir if not exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "{}_*epoch*.h5".format('unet'))
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")

    def load_weights(self,filepath):
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving


        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") else keras_model.layers

        try:
            saving.load_weights_from_hdf5_group(f, layers)
            print('load weight from %s'%filepath)

            if hasattr(f, 'close'):
                f.close()
        except:
            print('no weights to load')

    def build(self):
        if self.type == 0:
            return unet()
        else:
            return unet_res()

    def train(self,train_gene,val_gene,epochs):
        callbacks = [TensorBoard(log_dir='./logs', write_graph=True, write_grads=True),
                     ModelCheckpoint(self.checkpoint_path, monitor='loss', verbose=1,save_best_only=True),]

        self.keras_model.fit_generator(
            train_gene,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=500,
            callbacks=callbacks,
            validation_data=val_gene,
            validation_steps=100,
            max_queue_size=100,
            use_multiprocessing=False,
        )
        #self.epoch = max(self.epoch, epochs)

    def predict(self,val_generator,batch_size=30,verbose=1):
        return self.keras_model.predict_generator(val_generator,batch_size,verbose=1)