# demonstration of calculating metrics for a neural network model using sklearn
from sklearn.datasets import make_circles
from sklearn.datasets import load_sample_image
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_extraction import image
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# generate and prepare the dataset
def get_data():              
        # generate dataset
        X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
        #print("Which type you are ..?= am ",len(X),len(y))	
        #print(X)
        #print("-------------------")
        #print(y)
        # split into train and test
        n_test = 500
        trainX, testX = X[:n_test, :], X[n_test:, :]
        trainy, testy = y[:n_test], y[n_test:]
        one_image = load_sample_image("flower.jpg")
        print('Image shape: {}'.format(one_image.shape))
        patches = image.extract_patches_2d(one_image, (2, 2))
        print('Patches shape: {}'.format(patches.shape))
        print(patches[1])
        print(patches[800])
        return trainX, trainy, testX, testy

# define and fit the model
def get_model(trainX, trainy):
        # define model
        model = Sequential()
        model.add(Dense(100, input_dim=2, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit model
        model.fit(trainX, trainy, epochs=300, verbose=0)
        return model

# generate data
trainX, trainy, testX, testy = get_data()
# fit model
model = get_model(trainX, trainy)


# predict probabilities for test set
yhat_probs = model.predict(testX, verbose=0)
# predict crisp classes for test set
yhat_probs = model.predict(testX, verbose=0)
yhat_classes = yhat_probs.argmax(axis=-1)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
#yhat_classes = yhat_classes[:, 0]

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(testy, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(testy, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(testy, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(testy, yhat_classes)
print('F1 score: %f' % f1)

# kappa
kappa = cohen_kappa_score(testy, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(testy, yhat_probs)
print('ROC AUC: %f' % auc)
# confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
matrix = confusion_matrix(testy, yhat_classes)
sns.heatmap(matrix, annot=True, linewidths=0.01,cmap="Blues",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
print(matrix)
