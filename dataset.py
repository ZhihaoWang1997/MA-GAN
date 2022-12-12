
from PIL import Image
import cv2
import torch.utils.data as data
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        # CenterCrop(crop_size),
        Resize(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        # CenterCrop(crop_size),
        Resize(crop_size),
        ToTensor(),
    ])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    # img = Image.open(filepath).convert('YCbCr')
    # y, _, _ = img.split()
    # return y
    # image = cv2.imread(filepath)
    # return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.open(filepath)
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        # self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.image_filenames = image_dir

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input_image = load_img(self.image_filenames[index])
        target = input_image.copy()
        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.target_transform:
            target = self.target_transform(target)

        return input_image, target, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)

 
def get_dataset(upscale_factor, data_list):
    with open(data_list, mode='r') as f:
        dataset = f.readlines()
    dataset = [line.strip() for line in dataset]
    
    crop_size = calculate_valid_crop_size(224, upscale_factor)

    return DatasetFromFolder(dataset,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))

