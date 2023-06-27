import sys
import numpy as np

# Read the dataset from the text file
def read_dataset(filename):
    dataset = []
    with open(filename, 'r') as file:
        for line in file:
            binary_string = line.split()[0]
            dataset.append(binary_string)
    return dataset


# get files from user
testnet = sys.argv[1]
wnet = sys.argv[2]
dataset = read_dataset(testnet)
# Load the weights and the bias
with open(wnet, 'rb') as f1:
    a = np.load(f1)
    b = np.load(f1)
#print(a, b)

# Calculates the classification of the string according to weights and bias
def forward(X):
    weighted_sum = np.tanh(np.dot(a, X) + b)
    output = 1 if weighted_sum >= 0 else 0
    return output


# Creating a text file where each line contains
# the appropriate classification (the number 0 or 1) for this string.
with open('classification0.txt', 'w') as f:
    for s in dataset:
        # Calculates the classification of string s
        result = str(forward(np.array(list(s), dtype=int)))
        f.write(result + '\n')