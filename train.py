from utils.data import *
from utils.model import build_model
from tensorflow.keras.optimizers import Adam,SGD
from utils.loss_metric import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

EPOCHS = 300
BATCH_SIZE = 8
learning_rate = 1e-4
IMAGE_SIZE = 256

train_generator_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest')

train_gen = train_generator(data_training, BATCH_SIZE,
                                train_generator_args,
                                target_size=IMAGE_SIZE)
test_gener = train_generator(data_validation, BATCH_SIZE,
                                dict(),
                                target_size=IMAGE_SIZE)
                        
decay_rate = learning_rate / EPOCHS
opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)          

model = build_model((256,256, 3))

model.compile(optimizer=opt, loss=new_loss, metrics=[dice_coef,iou])             
callbacks = [ModelCheckpoint('a_seg1.hdf5',monitor='val_dice_coef',mode='max', verbose=1, save_best_only=True)] 

history = model.fit(train_gen,
                    steps_per_epoch=len(data_training) / BATCH_SIZE, 
                    epochs=EPOCHS,
                    callbacks = callbacks,
                    validation_data = test_gener,
                    validation_steps=len(data_validation) / BATCH_SIZE)

