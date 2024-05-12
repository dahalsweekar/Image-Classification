from tensorflow.keras import optimizers


class Train:
    def __init__(self, model, X_train, y_train, epoch):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.epoch = epoch

    def train(self):
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()
        self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=32, validation_split=0.2)
        self.model.save('./model/my_model.h5')
        print('Model saved.')
