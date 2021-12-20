from random import randint, choice
import random
import numpy as np
import argparse
import wandb
import os
import yaml

import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torchvision import transforms

from adamp import AdamP

from easydict import EasyDict

from dalle_pytorch import VQGanVAE
from dalle_pytorch.vae import VQGanVAE


from loader import TextImageDataset, ImgDatasetExample
from dalle.models import DALLE_Klue_Roberta
from utils import set_seed


def save_model(save_path, params, model):
    save_obj = {"hparams": params, "vae_params": None, "weights": model.state_dict()}
    torch.save(save_obj, save_path)


def train():
    for epoch in range(DALLE_CFG.EPOCHS):
        for i, (text, images, mask) in enumerate(dl):
            text, images, mask = map(lambda t: t.to(device), (text, images, mask))

            loss = dalle(text, images, mask=mask, return_loss=True)

            loss.backward()
            clip_grad_norm_(dalle.parameters(), DALLE_CFG.GRAD_CLIP_NORM)

            opt.step()
            opt.zero_grad()

            log = {}

            if i % 100 == 0:
                print(epoch, i, f"loss - {loss.item()}")

                log = {**log, "epoch": epoch, "iter": i, "loss": loss.item()}

            if i % 200 == 0:
                sample_text = text[:1]
                token_list = sample_text.masked_select(sample_text != 0).tolist()
                decoded_text = tokenizer.decode(token_list)

                image = dalle.generate_images(
                    text[:1], mask=mask[:1], filter_thres=0.9  # topk sampling at 0.9
                )
                save_model(f"{args.save_path}/dalle_uk.pt", dalle_params, dalle)
                wandb.save(f"{args.save_path}/dalle_uk.pt")

                log = {**log, "image": wandb.Image(image, caption=decoded_text)}

            wandb.log(log)

        # save trained model to wandb as an artifact every epoch's end

        model_artifact = wandb.Artifact(
            "trained-dalle", type="model", metadata=dict(dalle_params)
        )
        model_artifact.add_file(f"{args.save_path}/dalle_uk.pt")
        run.log_artifact(model_artifact)


if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_folder",
        type=str,
        default="/opt/ml/DALLE-Couture/data/cropped_train_img",
        help="",
    )
    parser.add_argument(
        "--text_folder", type=str, default="/opt/ml/DALLE-Couture/data/train_label",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="")
    parser.add_argument(
        "--transformer",
        type=str,
        default="basic",
        help="Category of image transformer.",
    )
    parser.add_argument(
        "--wte", type=str, default="/opt/ml/DALLE-pytorch/roberta_large_wte.pt", help=""
    )
    parser.add_argument(
        "--wpe", type=str, default="/opt/ml/DALLE-pytorch/roberta_large_wpe.pt", help=""
    )
    parser.add_argument(
        "--save_path", type=str, default="./results", help="save dalle model path"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="no_name",
        help="Name to save in the wandb log.",
    )
    parser.add_argument(
        "--vae_config",
        type=str,
        default="/opt/ml/KoDALLE/configs/vae_config.yaml",
        help="",
    )
    parser.add_argument(
        "--dalle_config",
        type=str,
        default="/opt/ml/KoDALLE/configs/dalle_config.yaml",
        help="",
    )

    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # Configuration
    with open(args.vae_config, "r") as f:
        vae_config = yaml.load(f)
        VAE_CFG = EasyDict(vae_config["VAE_CFG"])

    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")  # Korean Tokenizer
    with open(args.dalle_config, "r") as f:
        dalle_config = yaml.load(f)
        DALLE_CFG = EasyDict(dalle_config["DALLE_CFG"])
    DALLE_CFG.VOCAB_SIZE = tokenizer.vocab_size

    vae = VQGanVAE(VAE_CFG.MODEL_PATH, VAE_CFG.CONFIG_PATH)
    DALLE_CFG.IMAGE_SIZE = vae.image_size

    dalle_params = dict(
        num_text_tokens=tokenizer.vocab_size,
        text_seq_len=DALLE_CFG.TEXT_SEQ_LEN,
        depth=DALLE_CFG.DEPTH,
        heads=DALLE_CFG.HEADS,
        dim_head=DALLE_CFG.DIM_HEAD,
        reversible=DALLE_CFG.REVERSIBLE,
        loss_img_weight=DALLE_CFG.LOSS_IMG_WEIGHT,
        attn_types=DALLE_CFG.ATTN_TYPES,
        ff_dropout=DALLE_CFG.FF_DROPOUT,
        attn_dropout=DALLE_CFG.ATTN_DROPOUT,
        stable=DALLE_CFG.STABLE,
        shift_tokens=DALLE_CFG.SHIFT_TOKENS,
        rotary_emb=DALLE_CFG.ROTARY_EMB,
    )

    # Image Dataset
    initial_transformation = transforms.Compose(
        [
            transforms.Lambda(
                lambda img: img.convert("RGB") if img.mode != "RGB" else img
            ),
            transforms.Resize([VAE_CFG.IMAGE_SIZE, VAE_CFG.IMAGE_SIZE]),
            # transforms.CenterCrop(VAE_CFG.IMAGE_SIZE),
            transforms.ToTensor(),
        ]
    )

    dataset_visual = ImgDatasetExample(
        image_folder=args.image_folder, image_transform=initial_transformation
    )

    dataloader_visual = DataLoader(
        dataset=dataset_visual, batch_size=args.batch_size, shuffle=True
    )

    # Text to Image Dataset
    ds = TextImageDataset(
        text_folder=args.text_folder,
        image_folder=args.image_folder,
        text_len=DALLE_CFG.TEXT_SEQ_LEN,
        image_size=DALLE_CFG.IMAGE_SIZE,
        resize_ratio=DALLE_CFG.resize_ratio,
        truncate_captions=DALLE_CFG.truncate_captions,
        tokenizer=tokenizer,
        shuffle=True,
    )
    assert len(ds) > 0, "dataset is empty"

    dl = DataLoader(ds, batch_size=DALLE_CFG.BATCH_SIZE, shuffle=True, drop_last=True)

    # DALLE Model
    dalle = DALLE_Klue_Roberta(
        vae=vae, wpe_dir=args.wpe, wte_dir=args.wte, **dalle_params,
    ).to(device)
    opt = AdamP(dalle.parameters(), lr=DALLE_CFG.LEARNING_RATE)

    # Wandb
    run = wandb.init(
        project="optimization",
        entity="happyface-boostcamp",
        resume=False,
        config=dalle_params,
        name=args.wandb_name,  # change it when you experiment
    )

    train()
