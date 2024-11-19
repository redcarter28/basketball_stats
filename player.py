from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from colorama import Fore
import os

class Player:
    def __init__(self, name, props, team, model, stats, y_test, y_pred):
        self.name = name
        #self.bday = bday
        self.props = props
        self.team = team
        self.model = model
        self.stats = stats
        self.y_test = y_test
        self.y_pred = y_pred
        
    def accuracy_service(self, test_size):

        os.system('cls')
        print(Fore.LIGHTBLUE_EX + "Player Model Accuracy Report for: " + Fore.WHITE + f'{self.name}')

        print(Fore.YELLOW + f'This report was generated with a test_size of {test_size}, meaning %{str(test_size).split('.')[1] + '0'} of the data was reserved for test cases and the rest is used for training.\n')
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(Fore.WHITE + f'Accuracy: {accuracy:.2f}')

        print('Classification Report:')
        print(classification_report(self.y_test, self.y_pred))

        print('Confusion Matrix:')
        print(confusion_matrix(self.y_test, self.y_pred))
        data = input('\nPress any key to exit!')
        return