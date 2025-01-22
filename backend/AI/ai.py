import pandas as pd
from Classifier import Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import numpy as np

mnist = pd.read_csv("../../mnist.csv", header = None)
tags = mnist.pop(0)
images = mnist / 255

x_train, x_test, y_train, y_test = train_test_split(images, tags, test_size = 5000)

model = Classifier(
    sizes = [784, 16, 10], 
    learning_rate = 0.1,
    batch_size = 32,
    epochs = 50
)

model.fit(x_train, y_train)

model.to_json("../web/models","numberClassifier.json")

prediction = model.predict(x_test)
print(accuracy_score(y_test, prediction))

# one predict only
# a = images.iloc[1]
# prediction = model.getOutput(a.values.reshape(-1, 1))
# print("The number is: ", tags[1])
# print("The prediction is: ", np.argmax(prediction))

