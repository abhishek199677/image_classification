import tensorflow as tf 
from tensorflow.keras import layers , models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# hyperparameters 
IMG_SIZE = (128,128)
BATCH_SIZE = 16

# data preprocessing
#both train_datagen and val_datagen will rescale the images from RGB to black and white
train_datagen  = ImageDataGenerator(rescale= 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)

#reading the files from the image folder
train_data = train_datagen.flow_from_directory(    #train_data
    'data/train',
    target_size =IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical'
)
val_data = val_datagen.flow_from_directory(     #validation_data
    'data/val',
    target_size =IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical'
)


num_class = len(train_data.class_indices) # shows the total number of classes 

#buildimg the image model architecture
model =models.Sequential([
    layers.Conv2D(32 ,(3,3),activation = 'relu' ,input_shape = (128,128,3) ,padding ="same"),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(64 ,(3,3),activation = 'relu' ,padding ="same" ),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(128 ,(3,3),activation = 'relu',padding ="same" ),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(64 ,(3,3),activation = 'relu' ,padding ="same" ),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(254 ,(3,3),activation = 'relu' ,padding ="same" ),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(128 ,(3,3),activation = 'relu' ,padding ="same"),
    layers.MaxPooling2D(2,2),
    
    layers.Flatten(),
    layers.Dense(128,activation = 'relu'),
    layers.Dropout(.3),
    layers.Dense(num_class , activation = 'softmax')
]   
)

# this is to compile the model 
model.compile(
    loss = "categorical_crossentropy",
    optimizer = 'adam',
    metrics = ['accuracy']
    
)

# training the model 
model.fit(
    train_data,
    epochs = 10 , 
    validation_data = val_data
)


# saving the model
model.save("image_cnn_classifier.h5")

model.summary()

9989253841...........! 