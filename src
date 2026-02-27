train_dataset  = tf.keras.utils.image_dataset_from_directory(
    directory = '/content/training_images',
    image_size = (224,224),
    batch_size = 3,
    label_mode = 'categorical'
)
AUTOTUNE  = tf.data.AUTOTUNE
train_dataset_final = train_dataset.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)

#USING THE VGG16 FOR TRANSFER LEARNING
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense,Flatten, GlobalAveragePooling2D

def feature_extractor(inputs):
  feature_extractor_layer = tf.keras.applications.vgg16.VGG16(
      input_shape=(224, 224, 3),
      include_top=False,
      weights='imagenet')(inputs)

  return feature_extractor_layer

# ADDING FULLY CONNECTED LAYERS SUITED TO THE CLASSIFICATION PROBLEM
def classifier(inputs):

  x = GlobalAveragePooling2D()(inputs)
  x = Flatten()(x)
  x = Dense(1024, activation  = 'relu')(x)
  x = Dense(512, activation = 'relu')(x)
  x = Dense(10, activation = 'softmax')(x)

  return x

# STITCH THE TWO PARTS TO FORM THE FINAL MODEL

def final_model(inputs):

  vgg16_feature_extractor  = feature_extractor(inputs)
  image_classifier = classifier(vgg16_feature_extractor)

  return image_classifier
def compile_model():
  return final_model.compile(
      loss = 'CategoricalCrossentropy',
      optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.001),
      metrics = ['accuracy']
  )
def fit_model():
    return final_model.fit(
        train_dataset_final,
        verbose = False
    )
    
compile_model()
fit_model()
