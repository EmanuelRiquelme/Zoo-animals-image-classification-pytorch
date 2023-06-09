import torch
import os
from  PIL import Image
from torchvision import transforms

transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize((192,192)),
        transforms.RandomHorizontalFlip(p=.3),
        transforms.Normalize(mean = (.5,.5,.5),std = (.5,.5,.5)),
                ])

def clean(root_dir = 'animals',transform = transform):
    files = []
    for sub_dir in os.listdir(root_dir):
        files.extend([f'{root_dir}/{sub_dir}/test/{file}' for file in os.listdir(f'{root_dir}/{sub_dir}/test')])
    for sub_dir in os.listdir(root_dir):
        files.extend([f'{root_dir}/{sub_dir}/train/{file}' for file in os.listdir(f'{root_dir}/{sub_dir}/train')])
    files = [file for file in files if not file.endswith('.xml')]
    for file in files:
        try:
            transform(Image.open(file))
        except:
            print(file)
            os.remove(file)

if __name__ == '__main__':
    clean()
