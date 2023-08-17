# Import package to run AIML
import aiml

class ruleBasedChecker:
    
    def __init__(self):
        
        # Instantiate the Kernel -- allows for use for AIML file
        self.kernel = aiml.Kernel()
        self.kernel.setTextEncoding(False)
        
        # Learn patterns from the given aiml file stored within an xml file
        self.kernel.bootstrap(learnFiles='inputs/chatbot_rules.xml')
    
    def respond(self, statement):
        
        # Get the response from the kernel
        return self.kernel.respond(statement)
