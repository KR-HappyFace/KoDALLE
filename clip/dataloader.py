from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import re


class CLIPDataset(Dataset):
    def __init__(self, texts, images):
        self.texts = texts
        self.images = images
        self.transform = A.Compose([A.Resize(224, 224), ToTensorV2()])

    def __getitem__(self, index):
        t = self.texts[index]
        single_im = cv2.imread(self.images[index])
        single_im = cv2.cvtColor(single_im, cv2.COLOR_BGR2RGB)
        im = self.transform(image=single_im)["image"]
        return t, im

    def __len__(self):
        return len(self.texts)


def get_dataset(text_path, image_path):
    image_files = [
        *image_path.glob("**/*[0-9].png"),
        *image_path.glob("**/*[0-9].jpg"),
        *image_path.glob("**/*[0-9].jpeg"),
    ]
    text_files = [*text_path.glob("**/*[0-9].txt")]
    texts = []
    print("Extracting text information!")
    for i in tqdm(range(len(text_files))):
        with open(text_files[i], "r", encoding="utf-8") as f:
            te = f.read()
            te = re.sub("스타일에서 스타일은 [가-힣]+.", "", te)
            te = re.sub("에서", "", te)
            texts.append(te)
    return texts, image_files
