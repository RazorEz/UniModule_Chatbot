import nltk
from nltk.sem import Expression
from nltk.inference import TableauProver, ResolutionProver
read_expr = Expression.fromstring
import pandas as pd
from typing import Tuple, List

class FirstOrderLogic:
    """Class which applied First Order Logic to phrases using a supplied knowledge base
    
    """
    
    def __init__(self):
        """Instantiate class

        Args:
            knowledge_base_path (str): The file path to the knowledge base file
        """
        
        # Read in the knowledge base
        knowledge: pd.DataFrame = pd.read_csv("inputs/knowledge_base.csv", header=None)

        self.knowledge_base = []    
        
        # Insert knowledge into the knowledge base using the file given
        for row in knowledge[0]:  
            self.knowledge_base.append(read_expr(row))

    
    def parse_statement(self, statement:str, start_ignore: int = 0) -> Tuple[str, List[str], str]:
        """Converts a statement into first order logic

        Args:
            statement (str): The statement to be parsed
            start_ignore (int): The number of words to ignore beginning of statement
        
        Returns:
            str: A string in first order logic format. predicate(subject) or predicate(subject_1, subject_2)
            list: A list of the subjects referred to within the statement
            str: The predicate found within the statement
        """
        
        # List of subjects referenced within statement
        subjectsList = []
        
        # Split the statement into a list of strings
        statement_split = statement.lower().split()
        
        # Check if 'not' is within the statement and remove it
        # this will make the fol statement -predicate(subject)
        is_negative = 'not' in statement_split
        
        if is_negative:
            statement_split.remove('not')
        
        # Apply Positional Tagging excluding the 'Check that' portion of the statement
        statement_tagged = nltk.pos_tag(statement_split[start_ignore:])
        print(statement_tagged)  
        
        # Retrieve only singular or plural nouns (NN/NNS) from sentence
        info = [elem[0] for elem in statement_tagged if elem[1] in ['NN', 'NNS', 'JJ', 'JJS']]
        print(info)
        
        # If the number of nouns within info is greater than 2, the statement is a multi-valued predicate
        # Format [subject, predicate, subject_2] -> predicate(subject_1, subject_2) & predicate(subject_1, subject_n) etc.
        
        # Minimum length needed to for statement in first order logic
        minimum_info = 2
        
        # If the information is a single-valued predicate
        if len(info) == minimum_info:
            
            # Get predicate
            predicate = info[1]
            
            # Get subject
            subject = info[0]
            
            subjectsList.append(subject)
            
            # Convert to first order logic format
            statement_first_order_logic = f'-{predicate}({subject})' if is_negative else f"{predicate}({subject})"
        
        # If the information is a multi-valued predicate
        elif len(info) == 3:
            
            # Get predicate
            predicate = info[1]
            
            # Statement has two subjects, so convert to 2 valued predicate
            subject_1 = info[0]
            subject_2 = info[2]
            
            subjectsList.append(subject_1)
            subjectsList.append(subject_2)
            
            # Convert to first order logic format
            statement_first_order_logic = f'-{predicate}({subject_1}, {subject_2})' if is_negative else f"{predicate}({subject_1}, {subject_2})"
        
        # FURTHER IMPROVEMENT If there are more than two subjects within the statement, loop through each possible combination
        
        return statement_first_order_logic, subjectsList, predicate
            
    def parse_multi_valued(self, subject: str, statement: list) -> str:
        
        #print(subject)
        
        #if statement[0] == 'not':
        #    
        #    negative = True
        #    
        #else:
        #    
        #    negative = False
        
        tagged = nltk.pos_tag(statement)
        print(tagged)
        
        predicate = [elem[0] for elem in tagged[:-1] if elem[1] in ['VBZ', 'JJR', 'NN', 'JJ']]
        #print(predicate)
        
        #if len(predicate) > 1:
        predicate = predicate[0]
        #else:
            #predicate = predicate    
        
        
        #predicate = predicate[0]
        print(predicate)
        second_subject = tagged[-1][0]
        
        fol_statement = f'{predicate}({subject}, {second_subject})'
        
        return fol_statement
        #return f'-{fol_statement}' if negative else fol_statement
        
    def check_knowledge(self, statement:str) -> str:
        """Proves whether the statement is true or false using a knowledge base

        EXTRA FEATURE: 
            1. Make it mulit-valued predicate. i.e. Father(Ez, John), Ez is the Father of John

        IMPROVEMENTS:
            1. Add multiple predicates into the first order logic statement
        
        Args:
            statement (str): The statement to be proven, this in raw format: 
                'CHECK THAT X IS Y'
                        or 
                'CHECK THAT X CAN Y'
                        or
                'CHECK THAT X IS [THING] OF Y'

        Returns:
            str: The status of applying the proof
        """
        
        fol_statement, subjects, predicate = self.parse_statement(statement, start_ignore=2)

        #  Check if statement is true
        proof = ResolutionProver().prove(read_expr(fol_statement), self.knowledge_base, verbose=False)

        if proof is False:
            
            # Check the opposite for contradiction
            opposite_proof = ResolutionProver().prove(read_expr(f"-{fol_statement}"), self.knowledge_base, verbose=False)
            
            # If opposite is False and actual is False, then the given must be completely new
            if (opposite_proof is False) & (proof is False):
                # The orginnal statement and contradiction is also false, so no knowledge is known
                return "I don't know anything about that"
            else: 
                # If only the opposite is True, then the statement is outright
                return "This is not true"
        else:
            
            if len(subjects) == 1:
                return f"This is true, {subjects[0].capitalize()} is a {predicate}"
            else:
                return f"This is true"
                

    def add_knowledge(self, statement: str) -> str:
        """Inserts the knowledge into in-memory knowledge base. Allows for multi-valued predicates

        Args:
            statement (str): The statement to be added, this in raw format: 'I KNOW THAT X IS Y'

        Returns:
            str: The status of adding the knowledge
        """

        fol_statement, subjects, predicate = self.parse_statement(statement, start_ignore=3)

        opposite_proof = ResolutionProver().prove(read_expr(f"-{fol_statement}"), self.knowledge_base, verbose=False)
        
        if opposite_proof is False:
            
            self.knowledge_base.append(read_expr(fol_statement))    
            #print(self.knowledge_base)
            
            if len(subjects) == 1:
                
                return f"Okay, I will remember that {subjects[0].capitalize()} is a {predicate}"    
                
            else:
                
                return f"Okay, I will remember that both {subjects[0].capitalize()} and {subjects[1].capitalize()} are {predicate}s"

        else:
            
            return f"The opposite of {fol_statement} is True, meaning your statement is false"
                
    def testing(self, fol: str):
        
        proof = ResolutionProver().prove(read_expr(f"{fol}"), self.knowledge_base, verbose=True)
        print(proof)
        
if __name__ == "__main__":
    
    kb = FirstOrderLogic()
    
    # Check if strawhat(ez) in kb, should return false
    print(kb.check_knowledge('Check that luffy is the captain of nami'))
    print(kb.check_knowledge('Check that nami is the captain of nami'))
    print(kb.check_knowledge('Check that luffy is not the captain of nami'))
    print(kb.check_knowledge('Check that luffy is not the captain of luffy'))
    
    print(kb.check_knowledge('Check that luffy is a pirate'))
    print(kb.check_knowledge('Check that luffy is not a pirate'))
    
    # Inserting knowledge
    print(kb.add_knowledge("I know that jinbe is a strawhat"))
    print(kb.check_knowledge("Check that jinbe is a pirate"))
    print(kb.check_knowledge("Check that jinbe is not a pirate"))
    print(kb.check_knowledge("Check that jinbe is bad"))
    print(kb.check_knowledge("Check that jinbe is good"))
    
    print(kb.check_knowledge("Check that luffy is the strongest"))
    print(kb.check_knowledge("Check that luffy is the weakest"))
    print(kb.check_knowledge("Check that nami is the weakest"))