import tensorflow as tf
import numpy as np
from PIL import Image
from utils.dataset import Load
from utils.model import Model
from utils.dataloader import DataLoader
from utils.train import Train
import argparse


def main():
    parser = argparse.ArgumentParser(description='Training Model')
    parser.add_argument('--demo', help='Run Demo', default=False, action='store_true')
    parser.add_argument('--train', help='Train model', default=False, action='store_true')
    parser.add_argument('--epoch', type=int, help='number of epochs', default=50)
    parser.add_argument('--img_path', help='path to image', default='./dataset/Cat/0.jpg')
    parser.add_argument('--data_path', help='root path to dataset',
                        default='./dataset')

    args = parser.parse_args()

    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    Demo(args['demo'], args['train'], args['epoch'], args['img_path'], args['data_path']).run()


class Demo:
    def __init__(self, demo, train, epoch, img_path, data_path):
        self.demo = demo
        self.train = train
        self.epoch = epoch
        self.img_path = img_path
        self.data_path = data_path

    def run(self):
        if self.demo:
            loaded_model = tf.keras.models.load_model('model/my_model.h5')
            img_file = self.img_path
            img = Image.open(img_file)
            img = img.resize((224, 224))
            img_rgb = Image.new('RGB', img.size)
            img_rgb.paste(img)
            img_array = np.array(img_rgb) / 255.0
            input_image = np.expand_dims(img_array, axis=0)
            predictions = loaded_model.predict(input_image)
            print(predictions)
            if int(predictions[0].item()) == 1:
                print("Dog")
            else:
                print("Cat")

        if self.train:
            root_dir = self.data_path
            images, labels = Load(root_dir=root_dir).load_dataset()
            X_train, X_test, y_train, y_test = DataLoader(images, labels, test_size=0.2, random_state=42).data_load()
            b_classification_model = Model().fit_model()
            Train(b_classification_model, X_train, y_train, self.epoch).train()


if __name__ == '__main__':
    main()
