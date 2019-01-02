from keras import layers, models, regularizers

x_input = layers.Input((100,))
hidden1 = layers.Dense(100, kernel_regularizer=regularizers.l2())(x_input)
hidden1 = layers.BatchNormalization()(hidden1)  # I think it's important
hidden1 = layers.ReLU()(hidden1)
hidden2 = layers.Dense(100, kernel_regularizer=regularizers.l2())(layers.add([x_input, hidden1]))
hidden2 = layers.BatchNormalization()(hidden2)
hidden2 = layers.ReLU()(hidden2)
hidden3 = layers.Dense(100, kernel_regularizer=regularizers.l2())(layers.add([x_input, hidden1, hidden2]))
hidden3 = layers.BatchNormalization()(hidden3)
hidden3 = layers.ReLU()(hidden3)
y = layers.Dense(100, activation='sigmoid', kernel_regularizer=regularizers.l2())(
    layers.add([x_input, hidden1, hidden2, hidden3]))

model = models.Model(x_input, y)
model.summary()
