import numpy as np

# Define the training data
training_data = [
    {"input": "hi", "intent": "greeting"},
    {"input": "hello", "intent": "greeting"},
    {"input": "how are you", "intent": "greeting"},
    {"input": "what can you do", "intent": "capabilities"},
    {"input": "tell me a joke", "intent": "joke"},
    {"input": "thanks", "intent": "gratitude"},
    {"input": "bye", "intent": "farewell"},
]

# Preprocess the training data
intents = []
words = []
for data in training_data:
    intent = data["intent"]
    sentence = data["input"].lower()
    intents.append(intent)
    words.extend(sentence.split())

words = sorted(list(set(words)))

# Create word-to-index and intent-to-index dictionaries
word_to_index = {word: i for i, word in enumerate(words)}
intent_to_index = {intent: i for i, intent in enumerate(set(intents))}

# Create training samples
training_samples = []
output_labels = []
for data in training_data:
    sentence = data["input"].lower()
    bag = [0] * len(words)
    for word in sentence.split():
        if word in words:
            bag[word_to_index[word]] = 1
    training_samples.append(bag)
    output_label = [0] * len(intent_to_index)
    output_label[intent_to_index[data["intent"]]] = 1
    output_labels.append(output_label)

# Convert training samples and output labels to numpy arrays
X = np.array(training_samples)
y = np.array(output_labels)

# Define the neural network architecture
class NeuralNetwork:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = 8
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, X):
        self.hidden = np.dot(X, self.weights1)
        self.hidden_activation = self.sigmoid(self.hidden)
        self.output = np.dot(self.hidden_activation, self.weights2)
        self.output_activation = self.sigmoid(self.output)
        return self.output_activation

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        self.hidden_error = self.output_delta.dot(self.weights2.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.hidden_activation)
        self.weights2 += self.hidden_activation.T.dot(self.output_delta)
        self.weights1 += X.T.dot(self.hidden_delta)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

# Initialize the neural network
input_size = len(words)
output_size = len(intent_to_index)
neural_network = NeuralNetwork(input_size, output_size)

# Train the neural network
neural_network.train(X, y, epochs=1000)

# Define a function to classify user input
def classify_input(input_text):
    bag = [0] * len(words)
    for word in input_text.lower().split():
        if word in words:
            bag[word_to_index[word]] = 1
    prediction = neural_network.forward(np.array([bag]))
    intent_index = np.argmax(prediction)
    intents_list = list(intent_to_index.keys())
    return intents_list[intent_index]

# ChatterBox loop
print("Chatterbox: Hi, how can I assist you?")
while True:
    user_input = input("User: ")
    intent = classify_input(user_input)
    if intent == "greeting":
        print("Chatterbox: Hello!")
    elif intent == "capabilities":
        print("Chatterbox: I can answer questions and tell jokes.")
    elif intent == "joke":
        print("Chatterbox: Why don't scientists trust atoms? Because they make up everything!")
    elif intent == "gratitude":
        print("Chatterbox: You're welcome!")
    elif intent == "farewell":
        print("Chatterbox: Goodbye!")
        break
    else:
        print("Chatterbox: Sorry, I didn't understand that.")
