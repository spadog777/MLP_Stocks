import pandas as pd
#data = pd.read_csv("Final_Train_Dataset.csv")
data = pd.read_csv("test.csv")
data = data[['company_name_encoded','experience', 'location', 'salary']]

#Cleaning the experience
exp = list(data.experience)
min_ex = []
max_ex = []

for i in range(len(exp)):
   exp[i] = exp[i].replace("yrs","").strip()
   min_ex.append(int(exp[i].split("-")[0].strip()))
   max_ex.append(int(exp[i].split("-")[1].strip()))

#Attaching the new experiences to the original dataset
data["minimum_exp"] = min_ex
data["maximum_exp"] = max_ex

#Label encoding location and salary
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['location'] = le.fit_transform(data['location'])
data['salary'] = le.fit_transform(data['salary'])

#Deleting the original experience column and reordering
data.drop(['experience'], inplace = True, axis = 1)
data = data[['company_name_encoded', 'location','minimum_exp', 'maximum_exp', 'salary']]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data[['company_name_encoded', 'location', 'minimum_exp', 'maximum_exp']] = sc.fit_transform(data[['company_name_encoded', 'location', 'minimum_exp', 'maximum_exp']])

#Splitting the dataset into  training and validation sets
from sklearn.model_selection import train_test_split
training_set, validation_set = train_test_split(data, test_size = 0.2, random_state = 21)

#classifying the predictors and target variables as X and Y
X_train = training_set.iloc[:,0:-1].values
Y_train = training_set.iloc[:,-1].values
X_val = validation_set.iloc[:,0:-1].values
y_val = validation_set.iloc[:,-1].values

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

#Importing MLPClassifier
from sklearn.neural_network import MLPClassifier

#Initializing the MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)

#Fitting the training data to the network
classifier.fit(X_train, Y_train)

#Predicting y for X_val
y_pred = classifier.predict(X_val)

#Importing Confusion Matrix
from sklearn.metrics import confusion_matrix
#Comparing the predictions against the actual observations in y_val
cm = confusion_matrix(y_pred, y_val)

#Printing the accuracy
print("Accuracy of MLPClassifier : ", accuracy(cm))


