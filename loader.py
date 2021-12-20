from random import randint
from pathlib import Path
from typing import Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from transformers import AutoTokenizer
from preprocess import remove_style, remove_subj

# -*- coding:utf8 -*-


class TextImageDataset(Dataset):
    def __init__(
        self,
        text_folder: str,
        image_folder: str,
        text_len: int,
        image_size: int,
        truncate_captions: bool,
        resize_ratio: float,
        tokenizer: AutoTokenizer = None,
        shuffle: bool = False,
    ) -> None:
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle
        # path = Path(folder)
        self.tokenizer = tokenizer
        text_path = Path(text_folder)
        text_files = [*text_path.glob("**/*[0-9].txt")]

        image_folder = image_folder
        image_path = Path(image_folder)
        image_files = [
            *image_path.glob("**/*[0-9].png"),
            *image_path.glob("**/*[0-9].jpg"),
            *image_path.glob("**/*[0-9].jpeg"),
        ]
        text_files = {text_file.stem: text_file for text_file in text_files}
        image_files = {image_file.stem: image_file for image_file in image_files}

        keys = image_files.keys() & text_files.keys()
        self.keys = list(keys)
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda img: img.convert("RGB") if img.mode != "RGB" else img
                ),
                transforms.Resize([image_size, image_size]),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, ind: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        key = self.keys[ind]
        text_file = self.text_files[key]
        image_file = self.image_files[key]

        image = Image.open(image_file)
        descriptions = text_file.read_text(encoding="utf-8")
        descriptions = remove_style(descriptions).split("\n")
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))

        # ADD PREPROCESSING FUNCTION HERE
        encoded_dict = self.tokenizer(
            descriptions,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.text_len,
            add_special_tokens=True,
            return_token_type_ids=False,  # for RoBERTa
        )

        # flattens nested 2D tensor into 1D tensor
        flattened_dict = {i: v.squeeze() for i, v in encoded_dict.items()}
        input_ids = flattened_dict["input_ids"]
        attention_mask = flattened_dict["attention_mask"]

        image_tensor = self.image_transform(image)
        return input_ids, image_tensor, attention_mask

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)


class ImgDatasetExample(Dataset):
    """only for baseline cropped images"""

    def __init__(
        self, image_folder: str, image_transform: transforms.Compose = None,
    ) -> None:

        self.image_transform = image_transform

        self.image_path = Path(image_folder)
        self.image_files = [
            *self.image_path.glob("**/*.png"),
            *self.image_path.glob("**/*.jpg"),
            *self.image_path.glob("**/*.jpeg"),
        ]

    def __getitem__(self, index: int) -> torch.tensor:
        image = Image.open(self.image_files[index])

        if self.image_transform:
            image = self.image_transform(image)
        return torch.tensor(image)

    def __len__(self) -> int:
        return len(self.image_files)
