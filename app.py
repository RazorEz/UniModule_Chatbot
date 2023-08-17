# Importing Pakcages For Graphical User Interface
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QMainWindow,
    QLineEdit, QVBoxLayout, QTextEdit, QHBoxLayout )
from PyQt6.QtCore import QSize

# Import Packages for functionality
import sys
from datetime import datetime

# Import Self Created Classes For Functionality
from chatbot_class import ChatBot

class ChatBotGUI(QMainWindow):
    """The graphical user interface to host the One Piece Chatbot

    Args:
        QMainWindow (PyQt6.QtWidgets.QMainWindow): The main window of the application
    """
    
    def __init__(self):
        """Class instantiation

        Args:
            bot (Chatbot): The class which performs all chatbot functionality
        """
        
        # Define the chatbot
        self.bot: ChatBot = ChatBot()
        
        ## Run init of QMainWindow
        super().__init__()
        
        # Set width and height of the user interface
        self.setFixedSize(QSize(500,500))
        
        # Set title of window
        self.setWindowTitle("One Piece Chatbot")
        
        # Create the layout to hold all widgets within the window
        self.widgetsLayout = QVBoxLayout()
        
        # Create the box to hold the user's input
        self.inputBox = QLineEdit()
        
        # Set the width and height of the input box
        self.inputBox.setFixedSize(QSize(400,50))
        
        # Create the box to hold the outputs from the user and the chatbot
        self.outputBox = QTextEdit()
        
        # Set output box to read-only, to prevent user from editing that chat logs
        self.outputBox.setReadOnly(True)
        
        # Create the button to submit the text from the user
        self.submitButton = QPushButton("Submit")
        
        # Allow the button to trigger the submitMessage function when clicked 
        self.submitButton.clicked.connect(self.submitMessage)
        
        # Create the button to export the chat log
        self.exportButton = QPushButton("Export")
        
        # Allow the button to trigger the exportChat button when clicked
        self.exportButton.clicked.connect(self.exportChat)
        
        # Organise Layout which holds the buttons and the user's input
        self.bottomLayout = QHBoxLayout()
        
        # Organise the layour to hold the buttons
        self.buttonsLayout = QVBoxLayout()
        self.buttonsLayout.setSpacing(0)
        
        # Insert the submit and export button into the buttons layout
        self.buttonsLayout.addWidget(self.submitButton)
        self.buttonsLayout.addWidget(self.exportButton)
        self.bottomLayout.addLayout(self.buttonsLayout)
        self.bottomLayout.addWidget(self.inputBox)
        
        # Insert the output box and the bottom layout to the window
        self.widgetsLayout.addWidget(self.outputBox)
        self.widgetsLayout.addLayout(self.bottomLayout)
        
        # Create a widget to hold the layout
        layout = QWidget()
        layout.setLayout(self.widgetsLayout)
        
        # Make the layout the main widget - show everything onto the screen
        self.setCentralWidget(layout)
        
        # Get the current date and time
        timeToLog = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        # Output the welcome message to the user
        self.outputBox.append(f"[{timeToLog}] BOT: Welcome to the One Piece ChatBot!")
        
    # Running Chat Bot
    def submitMessage(self):
        """Function to display a message to the output box
        """
        
        # Calcuate current date and time, as well as format it
        timeToLog:str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        # Get the current inputted text from the user
        inputted_text = self.inputBox.text()
        
        # If there is an input from the user
        if inputted_text != "":
            
            # Insert the current time and date as well as the message given by the user
            #self.outputBox.append(f"[{timeToLog}] YOU: {self.inputBox.text()}")
            self.outputBox.append(f"YOU: {self.inputBox.text()}")
            
            # Get the response from the chatbot
            bot_response = self.bot.getResponse(inputted_text)
            
            # Insert the response from the chatbot onto the output box
            #self.outputBox.append(f"[{timeToLog}] BOT: {bot_response}")
            self.outputBox.append(f"BOT: {bot_response}")
            
            # Empty input box once entered info
            self.inputBox.clear()
            
    # Exporting Chat
    def exportChat(self):
        """Function to export the chat logs to 'chat_logs.txt'
        """
        
        outputBox_text: str = self.outputBox.toPlainText()
        
        with open('outputs/chat_logs.txt', 'w+') as f:
            f.write(outputBox_text)

# Instantiate Application
app = QApplication(sys.argv)
 
# Instantiate GUI using the chatbot
window = ChatBotGUI()

# Widget are hidden by default, so needs to be shown
window.show()

# Quit the application once the window is closed
app.exec()

