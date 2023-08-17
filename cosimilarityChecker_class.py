import pandas as pd
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class CosimilarityChecker:
    
    def __init__(self):
        
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = self.__setup("inputs/questions_answer.csv")
        
    def __setup(self, file_path):
        
        self.q_and_a: pd.DataFrame = pd.read_csv(file_path)
        
        self.original_questions: list = self.q_and_a['question'].tolist()
        self.cleaned_questions = self.q_and_a['question'].apply(self.__clean_text)
        self.answers: list = self.q_and_a['answer'].tolist()
        
        vectorizer = TfidfVectorizer()

        # Fit vectoriser to questions and transform the data
        self.transformed_questions = vectorizer.fit_transform(self.cleaned_questions)

        return vectorizer
        
    def __clean_text(self, text: str) -> str:
        
        # Tokenize Text
        tokenised = word_tokenize(text)
        
        # Remove stop words
        filtered_text = [word.lower() for word in tokenised if ((word.lower() not in self.stop_words) & (word.lower() not in string.punctuation))]
        
        # Remove punctuation
        return " ".join(filtered_text)    
    
    def calculate_similarity(self, question) -> str:
        
        # Transform the given question
        transformed_question = self.vectorizer.transform([self.__clean_text(question)])

        # Calculate similarity between user's question and each question within the corpus
        similarity = cosine_similarity(transformed_question, self.transformed_questions)
        
        # Retrieve the highest similarity found and the its position in the list
        best_similarity = np.max(similarity[0])
        best_similarity_index = np.argmax(similarity[0])
        
        best_match: str = self.original_questions[best_similarity_index]
        best_answer: str = self.answers[best_similarity_index]
        
        if best_similarity >= 0.7:
            print(best_similarity)
            print(best_match)
            return best_answer
        else:
            return ""

        #return f"The best question match is {best_match}, the answer is {best_answer}"

if __name__ == "__main__":
    text_handler = CosimilarityChecker()
    print(text_handler.calculate_similarity("Who is One Piece?"))
