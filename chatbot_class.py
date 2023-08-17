# This file holds the functionality of the chatbot

import aiml 
import os 
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import regex as re
import joblib

# Rule Based Component
from ruleBasedChecker_class import ruleBasedChecker

# Neural Network Component
from neuralNetworkPredictor import OnePieceImageClassifier

# Cosimilarity Checking Component
from cosimilarityChecker_class import CosimilarityChecker

# First Order Logic Component
from firstOrderLogic_class import FirstOrderLogic

class ChatBot:
    """ChatBot specifically created for the domain of One Piece (Manga and Anime)
    """
    
    def __init__(self):
        
        # Instantiate the class used for returning result using set questions and answers
        self.rule_based_checker = ruleBasedChecker()
        
        # Instantiate the image classifier used for determining images
        self.image_classifier = OnePieceImageClassifier()
        
        # Instantiate the Question and Answer handler used for comparing sentence similarities
        self.q_and_a_handler = CosimilarityChecker()
        
        # Instante the logic inferencer, used for applying first order logic as well as created new statement
        self.logic_inferencer = FirstOrderLogic()
        
    def use_classifier(self, input: str) -> str:
        """Uses the classifier class to clasify an image using the pre-trained neural network

        Args:
            input (str): The input string given by the user

        Returns:
            None | str: Returns the result of classifying the image, if none are found then return None
        """
        
        # Check wether the string followed the correct format for a jpeg file name
        chosen_image = re.search(r"(\w+|\d+)+[.]jpg", input)
    
        # If there is a match using the regex pattern        
        if chosen_image is not None:
            
            # Get the image name from the input
            image_name = chosen_image.group()

            # Check file exists in inputs/images folder
            if os.path.exists(f"inputs/images/{image_name}"):
                
                # Get the path of the image
                image_path = f"inputs/images/{image_name}"
                
                # Make classification using the image path and return the result
                return self.image_classifier.predict(image_path)
            
            # If the file does not exists
            else:
                
                return "That image does not exist"   
        
        # If there is no match on the regex pattern
        else:
            
            # Return None as it is not possible to classify something that does not exist
            return ""
    
    def getResponse(self, input: str) -> str:
        
        answer = ""
                                            
        # How to determine what to use
        kernel_response = self.rule_based_checker.respond(input)
        
        #self.simplifyText(input)
    
        if kernel_response in ['CHECK', 'ADD']:
            
            # Use knowledge base
            print("Using knowledge base")
            
            if kernel_response == 'CHECK':
                
                answer = self.logic_inferencer.check_knowledge(input)
            
            else:
                
                answer = self.logic_inferencer.add_knowledge(input)
                
        else:
            
            # Check if image submitted
            answer = self.use_classifier(input)
            
            if answer == "":
                
                # Check Q & A / Cosine Similarity
                answer = self.q_and_a_handler.calculate_similarity(input)
                
                if answer == "":
                    
                    answer = kernel_response 

        if answer == "":
            
            answer = "I do not know"
                          
        return answer      
       
    def simplifyText(self, input: str) -> None:
        
        # Pos Tag the input
        tags = pos_tag(input.split())
        print(tags)
        
        #return tags        
