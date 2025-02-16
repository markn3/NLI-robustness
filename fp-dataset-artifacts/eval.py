import pandas as pd
import json
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib.pyplot as plt

def main_lines():
    # Data for noise probabilities, loss, and accuracy
    probabilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Base model
    accuracy = [0.7928542494773865, 0.7099959524264832, 0.635484516620636, 0.5645358562649482, 
                0.5118077939332944, 0.4697679281234741, 0.445397261243707, 0.4257939755169555, 
                0.4111360013484955, 0.4017711579799652]

    # # inoculated models trained on 20%
    # accuracy_inoculated_20 = [0.8219665884971619, 0.7754479050636292, 0.7298452854156494, 
    #                         0.6971701979637146, 0.6605364680290222, 0.6371132135391235, 
    #                         0.6165512800216675, 0.5914087891578674, 0.5825529098510742, 
    #                         0.5710504651069641]

    # # inoculated models trained on 40%
    # accuracy_inoculated_40 = [0.8053746223449707, 0.7659812569618225, 0.7247557044029236, 
    #                         0.6910626888275146, 0.674063503742218, 0.6425081491470337, 
    #                         0.6259161233901978, 0.6014861464500427, 0.5946661233901978, 
    #                         0.5955822467803955]

    # Create a line chart for Accuracy with overlay for base and inoculated models
    plt.figure(figsize=(10, 5))

    # Plot base model accuracy
    plt.plot(probabilities, accuracy, marker='o', linestyle='-', color='green', label='Base Model')

    # # Plot accuracy for inoculated model trained with 20% noise
    # plt.plot(probabilities, accuracy_inoculated_20, marker='o', linestyle='--', color='blue', label='Inoculated Model (20% Noise)')

    # # Plot accuracy for inoculated model trained with 40% noise
    # plt.plot(probabilities, accuracy_inoculated_40, marker='o', linestyle='-.', color='red', label='Inoculated Model (40% Noise)')

    # Labels and title
    plt.xlabel('Noise Probability')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Noise Probability (Combined Noise on Hypothesis and Premise)')
    plt.legend()
    plt.grid(True)
    plt.show()

def noise_lines():
    probabilities = [0.2, 0.4, 0.6, 0.8, 1.0]

    # Base model
    accuracy = [0.7716816067695618, 0.649633526802063, 0.5289087891578674, 0.4288477301597595, 
                0.3805985450744629]

    # inoculated models trained on 20%
    accuracy_inoculated_20 = [0.8125, 0.743179976940155, 0.669686496257782, 0.6130903959274292, 
                0.5516083240509033]

    # inoculated models trained on 40%
    accuracy_inoculated_40 = [0.7099959254264832, 0.5645358562469482, 0.4697679281234741, 0.42579397559165955, 
                0.4017711579799652]

    plt.figure(figsize=(10, 5))

    # Plot base model accuracy
    plt.plot(probabilities, accuracy, marker='o', linestyle='-', color='pink', label='Character-Level Noise')

    # Plot accuracy for inoculated model trained with 20% noise
    plt.plot(probabilities, accuracy_inoculated_20, marker='o', linestyle='--', color='purple', label='Word-Level Noise')

    # Plot accuracy for inoculated model trained with 40% noise
    plt.plot(probabilities, accuracy_inoculated_40, marker='o', linestyle='-.', color='orange', label='Combined Character and Word Noise')

    # Labels and title
    plt.xlabel('Noise Probability')
    plt.ylabel('Accuracy')
    plt.title('Effect of Noise Type on Accuracy Across Noise Probabilities')
    plt.legend()
    plt.grid(True)
    plt.show()

def sentence_lines():
    probabilities = [0.2, 0.4, 0.6, 0.8, 1.0]

    accuracy = [0.7992671132087708, 0.7304560542106628, 0.6930985450744629, 0.6587947607040405, 
                0.6472923159599304]

    # inoculated models trained on 20%
    accuracy_inoculated_20 = [0.7762622237205505, 0.6702972054481506, 0.5793973803520203, 0.5300285220146179, 
                0.5071253776550293]

    # inoculated models trained on 40%
    accuracy_inoculated_40 = [0.7099959254264832, 0.5645358562469482, 0.4697679281234741, 0.42579397559165955, 
                0.4017711579799652]

    plt.figure(figsize=(10, 5))

    # Plot base model accuracy
    plt.plot(probabilities, accuracy, marker='o', linestyle='-', color='green', label='Noise on Premise Only')

    # Plot accuracy for inoculated model trained with 20% noise
    plt.plot(probabilities, accuracy_inoculated_20, marker='o', linestyle='--', color='blue', label='Noise on Hypothesis Only')

    # Plot accuracy for inoculated model trained with 40% noise
    plt.plot(probabilities, accuracy_inoculated_40, marker='o', linestyle='-.', color='red', label='Noise on Both Premise and Hypothesis')

    plt.xlabel('Noise P robability')
    plt.ylabel('Accuracy')
    plt.title("Impact of Noise Location on Accuracy Across Noise Probabilities")
    plt.legend()
    plt.grid(True)
    plt.show()

# main_lines()
noise_lines()
# sentence_lines()