from keras import layers, models

x_input = layers.Input((100,))
hidden1 = layers.Dense(100, activation='relu')(x_input)
hidden2 = layers.Dense(100, activation='relu')(layers.add([x_input, hidden1]))
hidden3 = layers.Dense(100, activation='relu')(layers.add([x_input, hidden1, hidden2]))
y = layers.Dense(100, activation='sigmoid')(layers.add([x_input, hidden1, hidden2, hidden3]))

model = models.Model(x_input, y)
model.summary()
