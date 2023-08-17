# Import Preprocessing Stuff
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout, ZeroPadding2D, Rescaling, Reshape, MaxPooling2D, Resizing, InputLayer, AveragePooling2D
from keras.layers.core import Activation
import keras.backend as K
import tensorflow as tf
from PIL import Image
import numpy as np
import joblib
from keras.utils import to_categorical
from ann_visualizer.visualize import ann_viz
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

MODEL_NAME = 'one_piece_model'
X_PATH = "neural_network/train_X.npy"
Y_PATH = "neural_network/train_y.npy"
CLASS_NAMES_PATH = "neural_network/class_names.joblib"
MODEL_SAVE_PATH = "inputs/models"

# Modelling Parameter
NETWORK_LAYERS = [
    InputLayer(input_shape=(200,200,3)),
    #Resizing(64, 64),
    Conv2D(16, kernel_size=3, input_shape=(200,200,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(2, activation='softmax')
]
EPOCHS = 9

# WAYS TO IMPROVE # 
# 1 - Image Augmentation. Lacking a decent number of images

class OnePieceCNN:
    
    def __init__(self, x_path, y_path, class_names_path):
        
        self.x_path = x_path
        self.y_path = y_path
        self.class_names_path = class_names_path
        
    def load_data(self):
        
        print(f"Loading training data from {self.x_path}....")
        print(f"Loading training labels from {self.y_path}....")
        
        self.X: np.ndarray = np.load(self.x_path)
        y: np.ndarray = np.load(self.y_path)
        
        #print(y.shape)
        # Encode target variable
        self.y_encoded = to_categorical(y)
        
        self.class_names: list = joblib.load(self.class_names_path)
        #self.class_names = list(self.class_names_dict.values())
    
        print("------ DATA INFORMATION -----")
        print(f"Training Shape: {self.X.shape}")
        print(f"Labels Shape: {self.y_encoded.shape}")
        print(f"Classes: {self.class_names}\n")  
    
        print("Training data has been loaded")
    
    def train_model(self, validation_split=0.2, epochs=2):
        
        if hasattr(self, 'model'):
            
            print(f"Fitting model using {validation_split*100} percent for validation....")
            self.model.fit(self.X, self.y_encoded, validation_split=validation_split, epochs=epochs)
            print("Model has been fitted...")
            
        else:
            
            print("Model has not been compiled, run compile_model to fix this")
            
    def compile_model(self, layers=None):
        
        if (hasattr(self, 'X') & hasattr(self, 'y_encoded')):
            
            print("Compiling model...")
            
            self.model = Sequential()

            if layers is None:
                
                layers = [
                
                InputLayer(input_shape=(200,200,3)),
                
                Resizing(64, 64),
                
                Conv2D(32, kernel_size=3, padding='same', activation='relu'),
                
                Conv2D(16, kernel_size=3, padding='same', activation='relu'),

                MaxPooling2D(pool_size=2),
                Flatten(),
                
                Dense(len(self.class_names), activation='softmax')

                ]


            for layer in layers:
                self.model.add(layer)
            
            #print(self.model.summary())
            
            # Compile model
            # This takes 3 arguments, optimiser, loss and metrics
            self.model.compile(
                optimizer='adam', #This adjusts the learning rate during training which determines how fast optimal weights 
                loss='categorical_crossentropy', # output label is assigned one-hot category which is good for one-hot encoded outputs
                metrics=['accuracy']
            )
            
            print(self.model.summary())
            print("Model compilation is complete")
            
        else:

            print("You must load data using load_data first")
       
    def save_model(self, directory, model_name):
        
        # Convert to tensorflowlite model
        #print("Converting model to TFLite Format....")
        #converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        #tflite_model = converter.convert()
        
        # Save the model.
        #print("Saving Model....")
        #with open(f'{directory}/{model_name}.tflite', 'wb') as f:
        #  f.write(tflite_model)
          
        #print(f'Model has been saved in {directory} as {model_name}.tflite')
        
        # Save model using h5 format
        print("Saving model in h5 format")
        self.model.save(f"{directory}/{model_name}.h5")
        print(f'Model has been saved in {directory} as {model_name}.h5')
        #print(f'Class names can be found in {self.class_names_path}')

    def get_model(self):
        return self.model
    
    def hypertune_model(self, param_grid:dict):
        
        keras_model = KerasClassifier(build_fn=self.get_model)
        
        grid = GridSearchCV(estimator=keras_model, param_grid=param_grid, cv=2)
        
        #grid_result = grid.fit(self.X, self.y_encoded)
        
        #print(grid_result.best_score_)
        #print(grid_result.best_params_)
        #print(grid_result.best_estimator_)

cnn = OnePieceCNN(X_PATH, Y_PATH, CLASS_NAMES_PATH)
cnn.load_data()
cnn.compile_model(layers=NETWORK_LAYERS)
cnn.train_model(validation_split=0.2, epochs=EPOCHS)
cnn.save_model(MODEL_SAVE_PATH, MODEL_NAME)