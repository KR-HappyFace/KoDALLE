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


def train(train_paths, val_paths=None):
    device = torch.device("cuda")
    model = CLIPModel()
    model = model.to(device)
    text_path, image_path = train_paths
    train_texts, train_images = get_dataset(text_path, image_path)

    train_dataset = CLIPDataset(train_texts, train_images)
    train_dl = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

    if val_paths:
        val_texts, val_images = get_dataset(text_path, image_path)
        val_dataset = CLIPDataset(val_texts, val_images)
        val_dl = DataLoader(val_dataset, batch_size=8, shuffle=False)
    num_epochs = 100
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
    scheduler = ReduceLROnPlateau(optimizer, "min")
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
    total_vloss = int(1e9)
    scaler = torch.cuda.amp.GradScaler()

    wandb.init(project="KoCLIP", entity="happyface-boostcamp")
    wandb.run.name = "dev"

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
            logits = text_embeddings @ image_embeddings.T * 0.07
            targets = torch.arange(len(text_embeddings), device="cuda")

            texts_loss = nn.CrossEntropyLoss()(logits, targets)
            images_loss = nn.CrossEntropyLoss()(logits.T, targets.T)

            loss = (images_loss + texts_loss) / 2.0

            if (steps + 1) % 100 == 0 and steps > 100:
                print(f"Epoch {i}, step {steps+1}, Loss: {loss}")
                wandb.log({"Loss": loss.item()})
            epoch_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f"Epoch Loss: {epoch_loss / len(train_dl)}")
        # wandb.log({"Epoch": i, "Epoch Loss": epoch_loss / len(train_dl)})
        # vloss = evaluate(model, val_dl)
        # wandb.log({"Val Loss": vloss})
        # if vloss < total_vloss:
        #    total_vloss = vloss
        #    torch.save(model, "fashionclip_fn2.pt")
        #    print(f"Model saved. Current best val loss {total_vloss}")
        scheduler.step(epoch_loss)
        # print(f"Best Val Loss: {total_vloss}")
    # wandb.finish()
