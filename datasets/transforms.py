import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.25),
        A.Normalize(),
        ToTensorV2()
    ])

def get_val_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(),
        ToTensorV2()
    ])

def get_ssl_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=30, p=0.5),
        A.RandomResizedCrop(image_size, image_size, scale=(0.7, 1.0), p=0.5),
        A.ColorJitter(p=0.5),
        A.GaussNoise(var_limit=(5.0, 40.0), p=0.35),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])
