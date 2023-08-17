import tensorflow as tf
from tensorflow.lite.python.interpreter import SignatureRunner
from keras.models import load_model, Model
import numpy as np
from PIL import Image
import joblib

class OnePieceImageClassifier:
    """Class to applying image classification using a neural network
    """
        
    def __init__(self, model_path: str = None, class_names_path: str = None):
        """Instantiate the class

        Args:
            model_path (str): The file path to trained neural network 
            class_names_path (str): The file path to the class names used for the neural network
        """
        
        # Load the pre-trained .h5 model
        self.model: Model = load_model('inputs/models/one_piece_model.h5')
        
        # Get the class names the model has been trained with
        self.class_names: list = joblib.load('inputs/class_names.joblib')
        
        # Set the image height and width to resize images to
        self.img_height = 200
        self.img_width = 200

    def predict(self, image_path: str) -> str:
        """Classify an image found from the given image path

        Args:
            image_path (str): The file path of the image

        Returns:
            str: The result of classifying the image
        """
        
        # Load the image and resize to the correct width and height
        img = Image.open(image_path)
        img = img.resize((self.img_height, self.img_width))
        img = img.convert('RGB')
        
        # Convert to array
        img_array = np.asarray(img)
                
        # Add an extra dimension to the array so it is the correct shape that neural network requires
        img_array = tf.expand_dims(img_array, 0)
        
        # Make classification on the image using the model
        predictions = self.model.predict(img_array)
        evaluations = self.model.evaluate(img_array)
        #self.model(input_1=img_array)['dense']
        #print(predictions)
        #print(evaluations)
        #print(tf.nn.softmax(predictions))
        #print(tf.nn.softmax(np.array([0., 1.])))
        
        # Get the list of scores for each possible class name
        score = tf.nn.softmax(predictions)
        
        # Get the highest score i.e which class the image is most likelt to be
        best_score = np.max(score)

        # Convert to score to out of 100, rounded to 2 decimal places
        best_score = round(best_score * 100, 2)
        
        # Get the class name the image is most likely to be using the index of the greatest score
        predicted_class = self.class_names[np.argmax(score)]
        
        return f"I am {best_score}% sure this is an image of {predicted_class}"
        
        #return "This is a picture of {}. I am {:.2f} percent sure about it.".format(self.class_names[np.argmax(score)], 100 * np.max(score))


if __name__=="__main__":
    
    one_piece_classifier = OnePieceImageClassifier() 
    print(one_piece_classifier.predict("inputs/images/Luffy_1.jpg"))
    #one_piece_classifier.predict("inputs/images/Akainu_54.jpg")
