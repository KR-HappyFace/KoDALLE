import math
import os
import argparse
import yaml
import wandb

from math import sqrt
from pathlib import Path
from easydict import EasyDict

# torch
from PIL import Image
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

# vision imports
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from torchvision import transforms

# dalle classes and utils
from dalle_pytorch import DiscreteVAE

# dataset imports
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from loader import ImgDatasetExample
from utils import set_seed


def save_model(path):
    save_obj = {
        "hparams": vae_params,
    }
    save_obj = {**save_obj, "weights": vae.state_dict()}

    torch.save(save_obj, path)


def train():
    global_step = 0
    temp = dVAE_CFG.STARTING_TEMP

    for epoch in range(args.epochs):
        for i, images in enumerate(dl):
            images = images.to(device)

            loss, recons = vae(images, return_loss=True, return_recons=True, temp=temp)

            opt.zero_grad()
            loss.backward()
            opt.step()

            logs = {}

            if i % 100 == 0:
                print(epoch, i, f"loss - {loss.item()}")

                logs = {**logs, "epoch": epoch, "iter": i, "loss": loss.item()}

                k = args.num_images_save

                with torch.no_grad():
                    codes = vae.get_codebook_indices(images[:k])
                    hard_recons = vae.decode(codes)

                images, recons = map(lambda t: t[:k], (images, recons))
                images, recons, hard_recons, codes = map(
                    lambda t: t.detach().cpu(), (images, recons, hard_recons, codes)
                )
                images, recons, hard_recons = map(
                    lambda t: make_grid(
                        t.float(), nrow=int(sqrt(k)), normalize=True, range=(-1, 1)
                    ),
                    (images, recons, hard_recons),
                )

                logs = {
                    **logs,
                    "sample images": wandb.Image(images, caption="original images"),
                    "reconstructions": wandb.Image(recons, caption="reconstructions"),
                    "hard reconstructions": wandb.Image(
                        hard_recons, caption="hard reconstructions"
                    ),
                    "codebook_indices": wandb.Histogram(codes),
                    "temperature": temp,
                }

                wandb.save("./vae.pt")
                save_model(f"./vae.pt")

                # temperature anneal

                temp = max(
                    temp * math.exp(-dVAE_CFG.ANNEAL_RATE * global_step),
                    dVAE_CFG.TEMP_MIN,
                )

                # lr decay

                # Do not advance schedulers from `deepspeed_config`.
                sched.step()
            wandb.log(logs)
            global_step += 1

        model_artifact = wandb.Artifact(
            "trained-vae", type="model", metadata=dict(model_config)
        )
        model_artifact.add_file("vae.pt")
        run.log_artifact(model_artifact)

        save_model(os.path.join(args.save_path, f"vae_{epoch}.pt"))
    # save final vae and cleanup
    save_model("./vae-final.pt")
    wandb.save("./vae-final.pt")

    model_artifact = wandb.Artifact(
        "trained-vae", type="model", metadata=dict(model_config)
    )
    model_artifact.add_file("vae-final.pt")
    run.log_artifact(model_artifact)

    wandb.finish()


if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dvae_config",
        type=str,
        default="/opt/ml/KoDALLE/vqgan/config/dvae_config.yaml",
        help="",
    )
    parser.add_argument("--save_path", type=str, default="./results_vae")
    parser.add_argument(
        "--image_folder",
        type=str,
        default="/opt/ml/DALLE-Couture/data/cropped_train_img",
        help="path to your folder of images for learning the discrete VAE and its codebook",
    )
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=4.5e-06, help="learning rate"
    )
    parser.add_argument(
        "--lr_decay_rate", type=float, default=0.98, help="learning rate decay"
    )
    parser.add_argument(
        "--num_images_save", type=int, default=1, help="number of images to save"
    )

    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    with open(args.dvae_config, "r") as f:
        dvae_config = yaml.load(f)
        dVAE_CFG = EasyDict(dvae_config["dVAE_CFG"])

    # Image Dataset
    initial_transformation = transforms.Compose(
        [
            transforms.Lambda(
                lambda img: img.convert("RGB") if img.mode != "RGB" else img
            ),
            transforms.Resize([dVAE_CFG.IMAGE_SIZE, dVAE_CFG.IMAGE_SIZE]),
            # transforms.CenterCrop(VAE_CFG.IMAGE_SIZE),
            transforms.ToTensor(),
        ]
    )
    ds = ImgDatasetExample(
        image_folder=args.image_folder, image_transform=initial_transformation
    )
    dl = DataLoader(ds, args.batch_size, shuffle=True)

    vae_params = dict(
        image_size=dVAE_CFG.IMAGE_SIZE,
        num_layers=dVAE_CFG.NUM_LAYERS,
        num_tokens=dVAE_CFG.NUM_TOKENS,
        codebook_dim=dVAE_CFG.EMB_DIM,
        hidden_dim=dVAE_CFG.HID_DIM,
        num_resnet_blocks=dVAE_CFG.NUM_RESNET_BLOCKS,
    )

    vae = DiscreteVAE(
        **vae_params,
        smooth_l1_loss=dVAE_CFG.SMOOTH_L1_LOSS,
        kl_div_loss_weight=dVAE_CFG.KL_LOSS_WEIGHT,
    )
    vae = vae.to(device)
    assert len(ds) > 0, "folder does not contain any images"

    # optimizer
    opt = Adam(vae.parameters(), lr=args.learning_rate)
    sched = ExponentialLR(optimizer=opt, gamma=args.lr_decay_rate)

    model_config = dict(
        num_tokens=dVAE_CFG.NUM_TOKENS,
        smooth_l1_loss=dVAE_CFG.SMOOTH_L1_LOSS,
        num_resnet_blocks=dVAE_CFG.NUM_RESNET_BLOCKS,
        kl_loss_weight=dVAE_CFG.KL_LOSS_WEIGHT,
    )

    run = wandb.init(
        project="DALLE-Couture", entity="happyface-boostcamp", config=model_config,
    )

    train()

