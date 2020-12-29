from model import ResNet
from data import DataGenerator
from tensorflow.keras.layers import Layer

def main():
    data = DataGenerator()
    train_gen = data.train_data('path')
    valid_gen = data.test_data('/path')
    model = ResNet(3)
    print(ResNet(3).model().summary())
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    model.fit_generator(train_gen, validation_data=valid_gen, epochs=2)

if __name__ == '__main__':
    main()
