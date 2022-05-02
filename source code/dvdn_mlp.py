import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import shuffle
# Import MLPClassifer
from sklearn.neural_network import MLPClassifier
# Import train_test_split function
from sklearn.model_selection import train_test_split

# Load data
data=pd.read_csv('summary_dataset.csv')

data = shuffle(data)

print(data.head(10))

# Import LabelEncoder
from sklearn import preprocessing

# Creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
data['Symbol']=le.fit_transform(data['Symbol'])

# Spliting data into Feature and
X=data[
    ['Symbol', 'EPS', 'ROE', 'PE', 'PB', ]
]

y=data['APR']


# Split dataset into training set and test set
set_ratio = 0.15
set_ratio = 0.2
rand_state = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=set_ratio, random_state=rand_state)

'''
mlp = MLPClassifier(
    hidden_layer_sizes=(10,10),
    max_iter=20,
    alpha=1e-4,
    solver="sgd",
    verbose=10,
    random_state=1,
    learning_rate_init=0.01,
)

# this example won't converge because of resource usage constraints on
# our Continuous Integration infrastructure, so we catch the warning and
# ignore it here
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

fig, axes = plt.subplots(1, 10)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
print(vmin, vmax)
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    print(coef)
    ax.matshow(coef.reshape(5, 1), cmap=plt.cm.gray, vmin=0.5 * vmin, vmax=0.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()

'''


# Create model object
# 10, 10
ADJ_SIZE = 15
res_list = []
for size in range(1, ADJ_SIZE+1):
    #clf = MLPClassifier(hidden_layer_sizes=(size, ),
    clf = MLPClassifier(hidden_layer_sizes=(22, size),
                        random_state=5,
                        verbose=True,
                        learning_rate_init=0.01)

    # Fit data onto the model
    clf.fit(X_train,y_train)

    # Make prediction on test dataset
    ypred=clf.predict(X_test)

    # Import accuracy score
    from sklearn.metrics import accuracy_score

    # Calcuate accuracy
    res = accuracy_score(y_test,ypred)

    #print(res)
    res_list.append(res)

### parameters test
print(res_list)

### ploting

#x = np.linspace(-1, 1, 50)
x = list(range(1, 1+ADJ_SIZE))
y1 = res_list

plt.figure()
plt.plot(x, y1)

plt.xlabel("2-layer size")
plt.ylabel("accuracy")
plt.title("MLP training result")

#plt.show()
plt.savefig('visuals/2-layer_sizes.png')
