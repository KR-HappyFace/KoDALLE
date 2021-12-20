import torch.nn as nn
from transformers import AutoTokenizer, RobertaConfig, RobertaModel
import torch
import timm
import torch.nn.functional as F
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import DistilBertModel
import cv2
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class CustomDataset(Dataset):
    def __init__(self, texts, images):
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/distilkobert")
        # self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
        self.texts = texts
        self.images = images
        self.transform = A.Compose([A.Resize(224, 224), ToTensorV2()])

    def __getitem__(self, index):
        t = self.texts[index]
        single_im = cv2.imread(self.images[index])
        single_im = cv2.cvtColor(single_im, cv2.COLOR_BGR2RGB)
        # single_im = Image.open(images[index])
        im = self.transform(image=single_im)["image"]
        return t, im

    def __len__(self):
        return len(self.texts)


class ImageEncoder(nn.Module):
    def __init__(self, model_name, trainable):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=False, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model(x)


from transformers import DistilBertConfig


class TextEncoder(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        # if pretrained:
        # config = DistilBertConfig.from_pretrained("monologg/distilkobert")
        # config = RobertaConfig.from_pretrained("klue/roberta-large")
        # self.model = RobertaModel(config)
        # self.model = RobertaModel.from_pretrained("klue/roberta-large")
        # self.model = DistilBertModel(config)
        self.model = DistilBertModel.from_pretrained("monologg/distilkobert")

        for p in self.model.parameters():
            p.requires_grad = True
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(
            input_ids=input_ids,
            # token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIPModel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        # self.image_encoder = ModifiedResNet(
        #    [3, 4, 6, 3], 0, 2048
        self.image_encoder = ImageEncoder("resnet50", trainable=True)
        # self.image_encoder = ImageEncoder("xcit_tiny_12_p8_224_dist", trainable=True)
        self.text_encoder = TextEncoder(pretrained=True)
        # self.image_projection = ProjectionHead(
        #    embedding_dim=2048, projection_dim=268, dropout=0.1
        # )
        self.image_projection = ProjectionHead(
            embedding_dim=2048, projection_dim=512, dropout=0.1
        )
        self.text_projection = ProjectionHead(
            embedding_dim=768, projection_dim=512, dropout=0.1
        )
        self.temperature = 0.07

    def forward(self, text, image):
        # Getting Image and Text Features
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(
            input_ids=text["input_ids"],
            attention_mask=text["attention_mask"],
            # token_type_ids=text["token_type_ids"],
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        # logits = (text_embeddings @ image_embeddings.T) * self.temperature

        # targets = torch.arange(len(text_embeddings), device="cuda")
        # tttt = F.log_softmax(logits, dim=1)
        # texts_loss = nn.CrossEntropyLoss()(tttt, targets)
        # images_loss = nn.CrossEntropyLoss()(tttt.T, targets.T)

        # texts_loss = nn.CrossEntropyLoss()(logits, targets)
        # images_loss = nn.CrossEntropyLoss()(logits.T, targets.T)

        # images_similarity = image_embeddings @ image_embeddings.T
        # texts_similarity = text_embeddings @ text_embeddings.T
        # targets = F.softmax(
        #    (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        # )
        # targets = torch.arange(len(logits), device="cuda")
        # texts_loss = cross_entropy(logits, targets, reduction="none")
        # images_loss = cross_entropy(logits.T, targets.T, reduction="none")
        # texts_loss = nn.CrossEntropyLoss()(logits, targets)
        # images_loss = nn.CrossEntropyLoss()(logits.T, targets.T)
        # loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        return text_embeddings, image_embeddings
        # return loss.mean()
