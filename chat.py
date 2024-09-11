


# Import the libraries we need
import random  # For picking random responses
import json    # For working with JSON files
import torch   # For working with neural networks

# Import our neural network model and helper functions
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Set up to use GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Open and read our intents JSON file which has example questions and answers
with open('/Users/ameshajid/Documents/VisualStudioCode/Small Projects/Pytorch-Chatbot/intent.json', 'r') as json_data:
    intents = json.load(json_data)
# Load the saved model data from a file
FILE = "/Users/ameshajid/Documents/VisualStudioCode/Small Projects/Pytorch-Chatbot/data.pth"
#This change tells PyTorch to only load the model's weights and not any additional data that might execute code, making it safer.
data = torch.load(FILE, weights_only=True)

# Extract information needed to use the model
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Create the neural network model and load the saved data into it
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()  # Set the model to evaluation mode

# Name of the chatbot
bot_name = "Sam"

def get_response(msg):
    # This function will take a message as input and return a response based on what the model predicts.

    # Tokenize the sentence (split it into words)
    sentence = tokenize(msg)
    # `tokenize(msg)` breaks down the input message into individual words.

    # Convert the tokenized sentence into a format the model understands
    X = bag_of_words(sentence, all_words)
    # `bag_of_words(sentence, all_words)` turns these words into numbers that the model can work with.

    X = X.reshape(1, X.shape[0])  # Make sure it's the right shape
    # `X.reshape(1, X.shape[0])` makes sure the input data is in the correct shape (1 row and as many columns as needed).

    X = torch.from_numpy(X).to(device)  # Convert it to a tensor and move it to the device
    # `torch.from_numpy(X)` converts the data into a format PyTorch understands (a tensor).
    # `.to(device)` moves the data to the device where the model is (CPU or GPU).

    # Get the model's prediction
    output = model(X)
    # `model(X)` uses the model to make a prediction based on the input data.

    _, predicted = torch.max(output, dim=1)  # Get the index of the highest score
    # `torch.max(output, dim=1)` finds the highest score from the model's output and its index.
    # `predicted` is the index of the highest score, which represents the predicted tag.

    # Get the tag associated with the highest score
    tag = tags[predicted.item()]
    # `tags[predicted.item()]` gets the actual tag (like "greeting" or "goodbye") that matches the highest score.

    # Get the probabilities for each tag
    probs = torch.softmax(output, dim=1)
    # `torch.softmax(output, dim=1)` calculates the probability of each tag, turning scores into probabilities.

    prob = probs[0][predicted.item()]
    # `probs[0][predicted.item()]` gets the probability of the tag with the highest score.

    # If the probability is high enough, respond with a random response for the tag
    if prob.item() > 0.75:
        # If the probability is greater than 0.75 (indicating confidence), choose a response.
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
                # `random.choice(intent['responses'])` picks a random response from the list of responses for that tag.
    return "I do not understand"
    # If the probability is too low, return a default message saying "I do not understand".
