import torchvision
from torchvision import models
from torchvision import transforms
import torch
import torch.nn as nn
import os
import numpy as np


def get_model():
    model = models.vgg16(pretrained=True)
    model.classifier = nn.Sequential(*[model.classifier[i] for i in range(4)])
    return model


def get_preprocess_pipeline():
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return preprocess


def extract_features(image_path, output_path):
    print("[INFO] Instantiating Model")
    model = get_model()

    print("[INFO] Instantiating Preprocessing Pipeline")
    preprocess = get_preprocess_pipeline()

    print("[INFO] Instantiating Dataset Loader")
    data_set = torchvision.datasets.ImageFolder(
        root=image_path, transform=preprocess)
    data_loader = torch.utils.data.DataLoader(
        data_set, batch_size=1, shuffle=False)

    print("[INFO] Starting Feature Extraction")
    feature_list = list()
    filename_list = list()
    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            input_file = data_loader.dataset.samples[idx]
            features = model(data[0])
            feature_list.append(features.numpy()[0])
            filename_list.append(input_file[0])

            if idx % 1000 == 0:
                print("[INFO] Processing image {0} of {1}".format(idx, len(data_set)))

    print("[INFO] Saving Outputs")
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, "filenames.npy"), filename_list)
    np.save(os.path.join(output_path, "features.npy"), feature_list)
