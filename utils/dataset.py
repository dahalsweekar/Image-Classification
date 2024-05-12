import numpy as np
import os
from PIL import Image
from tqdm import tqdm


class Load:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load_dataset(self):
        images = []
        labels = []

        label_map = {'cat': 0, 'dog': 1}

        for label_name in label_map:
            label_dir = os.path.join(self.root_dir, label_name)
            label = label_map[label_name]

            filenames = [filename for filename in os.listdir(label_dir) if
                         filename.endswith(".jpg") or filename.endswith(".png")]
            for filename in tqdm(filenames, desc=f'Loading {label_name} images', unit='image'):
                img = Image.open(os.path.join(label_dir, filename))
                img = img.resize((500, 500))
                img_rgb = Image.new('RGB', img.size)
                img_rgb.paste(img)
                img_array = np.array(img_rgb) / 255.0
                images.append(img_array)
                labels.append(label_map[label_name])
            print(f'{len(images)}')
            print(f'{len(labels)}')
        return images, labels
