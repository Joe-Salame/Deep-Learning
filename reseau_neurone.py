import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#charger les données 
data = pd.read_csv('spam.csv')

#on enléve la colomne qui contient les données de prediction
x = data.drop("Spam", axis=1)
y = data["Spam"]

#Definir le model
model = Sequential()  

#Premiere couche qui prend 57 entrée et cree 12 neurones et utilise softmax comme fonction d'activation
model.add(Dense(20, input_dim=57, activation="softmax")) 
model.add(Dense(1, activation="sigmoid"))

#Compilation du modele en precisant la fonction loss 
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

#On commence a entrainer le réseau de neurones
history = model.fit(x,y,validation_split=0.33, epochs=150, batch_size=10)
print(history.history.keys())

#Tester les performances du réseau de neurones
_, accuracy = model.evaluate(x,y)
print("Model accuracy: %.2f"% (accuracy*100))

#Prédire la sortie de nouvelles données
predictions = model.predict(x)

#Arrondir la prédiction
rounded = [round(x[0]) for x in predictions]

#Graphe pour le model de precison
plt.plot(history.history['accuracy'],color="#A56EFF")
plt.plot(history.history['val_accuracy'],color="#4ED0CE")
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Graphe pour le model de perte
plt.plot(history.history['loss'],color="#A56EFF")
plt.plot(history.history['val_loss'],color="#4ED0CE")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()