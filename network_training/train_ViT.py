from __future__ import print_function

#transformer 
import os
import glob
import datetime, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import cv2
from keras.callbacks import TensorBoard
from utils_regressor_focal_dist import CustomModelCheckpoint


config = tf.compat.v1.ConfigProto()
config.gpu_options.visible_device_list= '0'
config.gpu_options.allow_growth = True
config.allow_soft_placement = False
config.log_device_placement = False 
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

#Save model 
model_name = 'model_multi_class/'
SAVE = "new_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'

output_folder = SAVE + model_name
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_log = output_folder + "Log/"
if not os.path.exists(output_log):
    os.makedirs(output_log)

output_weight = output_folder + "Best/"
if not os.path.exists(output_weight):
    os.makedirs(output_weight)

#Get paths
IMAGE_FILE_PATH_DISTORTED = ""

focal_start = 40
focal_end = 500
dist_end = 1.2
classes_focal = list(np.arange(focal_start, focal_end+1, 10))
classes_distortion = list(np.arange(0, 61, 1) / 50.)

def get_paths(IMAGE_FILE_PATH_DISTORTED):
    paths_train = glob.glob(IMAGE_FILE_PATH_DISTORTED + 'train/' + "*.jpg")
    paths_train.sort()
    parameters = []
    labels_focal_train = []
    for path in paths_train:
        curr_parameter = float((path.split('_f_'))[1].split('_d_')[0])
        labels_focal_train.append((curr_parameter - focal_start*1.) / (focal_end*1. - focal_start*1.)) #normalize bewteen 0 and 1
    labels_distortion_train = []
    for path in paths_train:
        curr_parameter = float((path.split('_d_'))[1].split('.jpg')[0])
        labels_distortion_train.append(curr_parameter/1.2)

    c = list(zip(paths_train, labels_focal_train,labels_distortion_train))
    random.shuffle(c)
    paths_train, labels_focal_train,labels_distortion_train = zip(*c)
    paths_train, labels_focal_train, labels_distortion_train = list(paths_train), list(labels_focal_train), list(labels_distortion_train)
    labels_train = [list(a) for a in zip(labels_focal_train, labels_distortion_train)]

    paths_valid = glob.glob(IMAGE_FILE_PATH_DISTORTED + 'valid/' + "*.jpg")
    paths_valid.sort()
    parameters = []
    labels_focal_valid = []
    for path in paths_valid:
        curr_parameter = float((path.split('_f_'))[1].split('_d_')[0])
        labels_focal_valid.append((curr_parameter-focal_start*1.)/(focal_end*1.+1.-focal_start*1.)) #normalize bewteen 0 and 1
    labels_distortion_valid = []
    for path in paths_valid:
        curr_parameter = float((path.split('_d_'))[1].split('.jpg')[0])
        labels_distortion_valid.append(curr_parameter/1.2)

    c = list(zip(paths_valid, labels_focal_valid, labels_distortion_valid))
    random.shuffle(c)
    paths_valid, labels_focal_valid, labels_distortion_valid = zip(*c)
    paths_valid, labels_focal_valid, labels_distortion_valid = list(paths_valid), list(labels_focal_valid), list(labels_distortion_valid)
    labels_valid = [list(a) for a in zip(labels_focal_valid, labels_distortion_valid)]


    return paths_train, labels_train, paths_valid, labels_valid

paths_train, labels_train, paths_valid, labels_valid = get_paths(IMAGE_FILE_PATH_DISTORTED)

print(len(paths_train), 'train samples')
print(len(paths_valid), 'valid samples')

input_shape = (299, 299, 3)

#Configure the hyperparameters
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 240
num_epochs = 10000
image_size = 300  # We'll resize input images to this size
patch_size = 50  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final

print(batch_size)

#Use data augmentation
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)

def load_images (paths_images):
    images = []
    for i in paths_images:
        image = np.array(cv2.imread(i))
        images.append(image)
    return images
        
x_train = np.asarray(load_images(paths_train))
y_train = np.array(labels_train)
x_valid = np.asarray(load_images(paths_valid))
y_valid = np.array(labels_valid)
#Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)

#Implement multilayer perceptron (MLP)
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

#Implement patch creation as a layer
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

#Implement the patch encoding layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

#Build the ViT model
def create_vit_regressor():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

    #logits = layers.Dense(num_classes)(features)
    final_output_focal = layers.Dense(1, name='output_focal')(features)
    final_output_distortion = layers.Dense(1, name='output_distortion')(features)
    # Create the Keras model.
    model = keras.Model(inputs, [final_output_focal, final_output_distortion])
    #adam = adam = tf.keras.optimizers.Adam(lr=learning_rate)
    #model.compile(loss={'output_focal':'logcosh', 'output_distortion':'logcosh'},
             # metrics={'output_focal':'logcosh','output_distortion':'logcosh'}
             # )
    #model.summary()
    return model

loss = tf.keras.losses.LogCosh()

#Compile and train
def run_train(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        loss={'output_focal':'logcosh', 'output_distortion':'logcosh'}, optimizer=optimizer,
                    metrics={'output_focal':'logcosh', 'output_distortion':'logcosh'}
    )
    model.summary()
    
    checkpoint_callback = CustomModelCheckpoint(
        model_for_saving=model,
        filepath=output_weight + "weights_{epoch:02d}_{loss:.2f}.h5",
        monitor="loss",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x = x_train,
        y = y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(x_valid,y_valid),
        callbacks=[tensorboard, checkpoint_callback],
        verbose=1
    )
    return history


vit_regressor = create_vit_regressor()

tensorboard = TensorBoard(log_dir=output_log)

history = run_train(vit_regressor)

model_json = vit_regressor.to_json()
with open(output_folder + "model.json", "w") as json_file:
    json_file.write(model_json)
