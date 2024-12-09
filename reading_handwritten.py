# importing the hand written digit dataset
from sklearn import datasets

#digit contain the dataset
digits = datasets.load_digits()

#dif function use to display the attributes of the dataset
dir(digits)

#digits.image is a three dimensional array.
#First indexes images, second two are pixels and xy coordinates
#each image is 8x8 =64

#outputting the picture value as series of numbers
print(digits.images[0])

#importing the matplotlib libraries pyplot function
import matplotlib.pyplot as plt

#defining the function plot_multi

def plot_multi(i):
    nplots=16
    fig = plt.figure(figsize=(15,15))
    for j in range(nplots):
        plt.subplot(4,4,j+1)
        plt.imshow(digits.images[i+j], cmap='binary')
        plt.title(digits.target[i+j])
        plt.axis('off')
    #printing the each digits in the dataset
    plt.show()
    
    plot_multi(0)
    
#convolution #maxpool etc..so simplifying data until
#you need to flatten them for input layer
    
#converting the 2D array to one dimensional array
y=digits.target
x=digits.images.reshape((len(digits.images),-1))
    
x.shape
# printing the one-dimensional array's values
x[0]

# Very first 1000 photographs and
#lables will be uised in training.

x_train = x[:1000]
y_train = y[:1000] 

#the leftover dataset will be utilised to 
#test the networks's performance later on.

x_test = x[1000:]
y_test = y[1000:]

#Importing Multilayer perceptron or MLP classifier
from sklearn.neural_network import MLPClassifier

#Calling the MLP classifier with specific parameters
mlp = MLPClassifier(hidden_layer_sizes=(15,),
                    activation='logistic',
                    alpha=1e-4, solver='sgd',
                    tol=1e-4, random_state=1,
                    learning_rate_init=.1,
                    verbose=True)

mlp.fit(x_train,y_train)

fig, axes = plt.subplots(1,1)
axes.plot(mlp.loss_curve_,'o-')
axes.set_xlabel('number of iteration')
axes.set_ylabel("loss")
plt.show()


#Testing
predictions=mlp.predict(x_test)
predictions[:70]

y_test[:50]

# importing the accuracy_score from the sklearn
from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions)

import joblib

# Save the trained model to a file
joblib.dump(mlp, 'mlp_digit_classifier.pkl')
