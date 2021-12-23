from torch.utils.data import DataLoader
from clipmodel import CLIPModel
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
import itertools
from dataloader import CLIPDataset, get_dataset
import argparse


def calculate_loss(text_embeds, image_embeds, temperature=0.07):
    logits = text_embeds @ image_embeds.T * temperature
    targets = torch.arange(len(text_embeds), device="cuda")

    texts_loss = nn.CrossEntropyLoss()(logits, targets)
    images_loss = nn.CrossEntropyLoss()(logits.T, targets.T)

    t_loss = (images_loss + texts_loss) / 2.0
    loss = t_loss.mean()
    return loss


def evaluate(model, val_dl, tokenizer):
    val_loss = 0
    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(tqdm(val_dl)):
            text, image = batch
            text = tokenizer(
                list(text),
                padding=True,
                pad_to_max_length=True,
                max_length=128,
                truncation=True,
                return_tensors="pt",
            )
            text = text.to(model.device)
            image = image.float()
            image = image.to(model.device)
            text_embeds, image_embeds = model(text, image)
            loss = calculate_loss(text_embeds, image_embeds)
            val_loss += loss
    print(f"Val Loss: {val_loss / len(val_dl)}")
    return val_loss / len(val_dl)


def get_optimizer(model):
    params = [
        {"params": model.image_encoder.parameters(), "lr": 4e-5},
        {"params": model.text_encoder.parameters(), "lr": 4e-5},
        {
            "params": itertools.chain(
                model.image_projection.parameters(), model.text_projection.parameters()
            ),
            "lr": 5e-5,
        },
    ]
    optimizer = AdamW(params, weight_decay=0.2, betas=(0.9, 0.98), eps=1e-6)
    return optimizer


def train(model, device, train_paths, val_paths=None, num_epochs=100):
    model = model.to(device)
    text_path, image_path = train_paths
    train_texts, train_images = get_dataset(text_path, image_path)

    train_dataset = CLIPDataset(train_texts, train_images)
    train_dl = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    if val_paths:
        val_texts, val_images = get_dataset(text_path, image_path)
        val_dataset = CLIPDataset(val_texts, val_images)
        val_dl = DataLoader(val_dataset, batch_size=8, shuffle=False)

    optimizer = get_optimizer(model)
    scheduler = ReduceLROnPlateau(optimizer, "min")
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
    total_vloss = int(1e9)
    scaler = torch.cuda.amp.GradScaler()

    for i in range(num_epochs):
        print(f"Epochs: {i+1}")
        epoch_loss = 0
        model.train()
        wandb.log({"Epochs": i + 1})
        for steps, batch in enumerate(tqdm(train_dl)):
            text, image = batch
            text = tokenizer(
                list(text),
                padding=True,
                pad_to_max_length=True,
                max_length=128,
                truncation=True,
                return_tensors="pt",
            )
            text = text.to(device)
            image = image.float()
            image = image.to(device)
            optimizer.zero_grad()

            text_embeddings, image_embeddings = model(text, image)
            loss = calculate_loss(text_embeddings, image_embeddings)

            if (steps + 1) % 100 == 0 and steps > 100:
                print(f"Epoch {i}, step {steps+1}, Loss: {loss.item()}")
            epoch_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f"Epoch Loss: {epoch_loss / len(train_dl)}")
        if val_paths:
            vloss = evaluate(model, val_dl, tokenizer)
            if vloss < total_vloss:
                total_vloss = vloss
                torch.save(model, "clip.pt")
                print(f"Model saved. Current best val loss {total_vloss}")
        scheduler.step(epoch_loss)


if __name__ == "__main__":
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_folder",
        type=str,
        default="/opt/ml/DALLE-Couture/data/cropped_train_img",
        help="",
    )
    parser.add_argument(
        "--text_folder",
        type=str,
        default="/opt/ml/DALLE-Couture/data/train_label",
    )
    args = parser.parse_args()
    model = CLIPModel()
    train_paths = [args.text_folder, args.image_folder]
    train(
        model,
        device,
        train_paths=train_paths,
    )
