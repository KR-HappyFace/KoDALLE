import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
import timm


class ImageEncoder(nn.Module):
    def __init__(self, model_name, pretrained):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model(x)


class TextEncoder(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        if pretrained:
            self.model = RobertaModel.from_pretrained("klue/roberta-base")
        else:
            config = RobertaConfig.from_pretrained("klue/roberta-base")
            self.model = RobertaModel(config)

        for p in self.model.parameters():
            p.requires_grad = True
        self.target_token_idx = 0

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        return projected


class CLIPModel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder("efficientnet_b0", pretrained=False)
        self.text_encoder = TextEncoder(pretrained=True)
        image_embedding_dim = list(self.image_encoder.parameters())[-1].shape[0]
        text_embedding_dim = list(self.text_encoder.parameters())[-1].shape[0]
        self.image_projection = ProjectionHead(
            embedding_dim=image_embedding_dim, projection_dim=512
        )
        self.text_projection = ProjectionHead(
            embedding_dim=text_embedding_dim, projection_dim=512
        )

    def forward(self, text, image):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(
            input_ids=text["input_ids"],
            attention_mask=text["attention_mask"],
            token_type_ids=text["token_type_ids"],
        )

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        return text_embeddings, image_embeddings
