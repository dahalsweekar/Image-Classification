from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models


class Model:
    def __init__(self):
        self.base_model = ResNet50(include_top=False, weights='imagenet') #, input_shape=(224, 224, 3))

    def fit_model(self):
        model = models.Sequential([
            self.base_model,
            layers.GlobalAvgPool2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

        return model
