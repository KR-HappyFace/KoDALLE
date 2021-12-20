from random import randint, choice
import random
import numpy as np
import argparse
import wandb
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms

from adamp import AdamP
from PIL import Image

from easydict import EasyDict
from pathlib import Path
import matplotlib.pyplot as plt

from dalle_pytorch import VQGanVAE, DALLE
from dalle_pytorch import distributed_utils, DiscreteVAE
from dalle_pytorch.vae import OpenAIDiscreteVAE, VQGanVAE
from dalle_pytorch.transformer import Transformer, DivideMax
from dalle_pytorch.attention import stable_softmax

from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange

# helpers
def exists(val):
    return val is not None


def set_seed(random_seed):
    """
    Random number fixed
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)


def vae_config() -> EasyDict:
    """Generator Model Configuration"""

    VAE_CFG = EasyDict()
    VAE_CFG.MODEL_PATH = (
        "/opt/ml/taming-transformers/logs/blue_e4/checkpoint/VQGAN_blue.ckpt"
    )
    VAE_CFG.CONFIG_PATH = (
        "/opt/ml/taming-transformers/logs/blue_e4/config/VQGAN_blue.yaml"
    )

    VAE_CFG.IMAGE_SIZE = 256
    VAE_CFG.IMAGE_PATH = "./"

    VAE_CFG.NUM_TOKENS = 16384  # codebook image patch tokens
    VAE_CFG.NUM_LAYERS = 3
    VAE_CFG.NUM_RESNET_BLOCKS = 2
    VAE_CFG.SMOOTH_L1_LOSS = False
    # VAE_CFG.EMB_DIM = 512
    # VAE_CFG.HID_DIM = 256
    VAE_CFG.KL_LOSS_WEIGHT = 0

    VAE_CFG.STARTING_TEMP = 1.0
    VAE_CFG.TEMP_MIN = 0.5
    VAE_CFG.ANNEAL_RATE = 1e-6

    VAE_CFG.NUM_IMAGES_SAVE = 4

    VAE_CFG.BASELINE = False
    return VAE_CFG


def dalle_config(tokenizer) -> EasyDict:
    """DALLE Configuration"""

    DALLE_CFG = EasyDict()
    DALLE_CFG.VOCAB_SIZE = (
        tokenizer.vocab_size
    )  # refer to EDA, there are only 333 words total. but input_ids index should be in within 0 ~ 52000: https://github.com/boostcampaitech2-happyface/DALLE-Couture/blob/pytorch-dalle/EDA.ipynb
    DALLE_CFG.DALLE_PATH = None  # './dalle.pt'
    DALLE_CFG.TAMING = True  # use VAE from taming transformers paper
    DALLE_CFG.BPE_PATH = None
    DALLE_CFG.RESUME = exists(DALLE_CFG.DALLE_PATH)

    DALLE_CFG.EPOCHS = 3
    DALLE_CFG.BATCH_SIZE = 16

    # configuration mimics of: https://github.com/lucidrains/DALLE-pytorch/discussions/131
    # Hyperparameter testing on pytorch-dalle: https://github.com/lucidrains/DALLE-pytorch/issues/84
    # Another Reference for Hyperparams https://github.com/lucidrains/DALLE-pytorch/issues/86#issue-832121328
    DALLE_CFG.LEARNING_RATE = 3e-4
    DALLE_CFG.GRAD_CLIP_NORM = 0.5

    DALLE_CFG.TEXT_SEQ_LEN = 64
    DALLE_CFG.DEPTH = 12
    DALLE_CFG.HEADS = 8
    DALLE_CFG.DIM_HEAD = 64  # 8개의 head, 64는 각 head의 dimension
    DALLE_CFG.REVERSIBLE = True
    DALLE_CFG.LOSS_IMG_WEIGHT = 7
    DALLE_CFG.ATTN_TYPES = "full"
    DALLE_CFG.FF_DROPOUT = 0.2  # Feed forward dropout
    DALLE_CFG.ATTN_DROPOUT = 0.2  # Attention Feed forward dropout
    DALLE_CFG.STABLE = None  # stable_softmax
    DALLE_CFG.SHIFT_TOKENS = None
    DALLE_CFG.ROTARY_EMB = None

    DALLE_CFG.resize_ratio = 1.0
    DALLE_CFG.truncate_captions = True
    return DALLE_CFG


class ImgDatasetExample(Dataset):
    """ only for baseline cropped images """

    def __init__(
        self, image_folder: str, image_transform: transforms.Compose = None,
    ):

        self.image_transform = image_transform

        self.image_path = Path(image_folder)
        self.image_files = [
            *self.image_path.glob("**/*.png"),
            *self.image_path.glob("**/*.jpg"),
            *self.image_path.glob("**/*.jpeg"),
        ]

    def __getitem__(self, index):
        image = Image.open(self.image_files[index])

        if self.image_transform:
            image = self.image_transform(image)
        return torch.tensor(image)

    def __len__(self):
        return len(self.image_files)


def basic_transformer(VAE_CFG: EasyDict) -> transforms.Compose:
    transformer = transforms.Compose(
        [
            transforms.Lambda(
                lambda img: img.convert("RGB") if img.mode != "RGB" else img
            ),
            transforms.Resize([VAE_CFG.IMAGE_SIZE, VAE_CFG.IMAGE_SIZE]),
            # transforms.CenterCrop(VAE_CFG.IMAGE_SIZE),
            transforms.ToTensor(),
        ]
    )
    return transformer


def show_transform(image, title="Default"):
    plt.figure(figsize=(16, 6))
    plt.suptitle(title, fontsize=16)

    # Unnormalize
    image = image / 2 + 0.5
    npimg = image.numpy()
    npimg = np.clip(npimg, 0.0, 1.0)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_transform_al(image, title="Default"):
    plt.figure(figsize=(16, 6))
    plt.suptitle(title, fontsize=16)

    # Unnormalize
    plt.imshow(image)
    plt.show()


# -*- coding:utf8 -*-
def remove_style(input_text: str) -> str:
    # split sentences by .
    sentences = input_text.split(".")
    return_sentences = []

    for sentence in sentences:
        if "스타일" in sentence:
            # remove the sentence from the list
            pass
        else:
            return_sentences.append(sentence)
            pass
    # join sentences into one str
    return ".".join(return_sentences).strip()


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
    ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle
        # path = Path(folder)
        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

        self.text_folder = text_folder
        self.text_path = Path(self.text_folder)
        self.text_files = [*self.text_path.glob("**/*.txt")]

        self.image_folder = image_folder
        self.image_path = Path(self.image_folder)
        self.image_files = [
            *self.image_path.glob("**/*.png"),
            *self.image_path.glob("**/*.jpg"),
            *self.image_path.glob("**/*.jpeg"),
        ]

        self.text_files = {text_file.stem: text_file for text_file in self.text_files}
        self.image_files = {
            image_file.stem: image_file for image_file in self.image_files
        }

        self.keys = self.image_files.keys() & self.text_files.keys()

        self.keys = list(self.keys)
        self.text_files = {k: v for k, v in self.text_files.items() if k in self.keys}
        self.image_files = {k: v for k, v in self.image_files.items() if k in self.keys}
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

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, ind):
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


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class always:
    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return self.val


def is_empty(t):
    return t.nelement() == 0


def masked_mean(t, mask, dim=1):
    t = t.masked_fill(~mask[:, :, None], 0.0)
    return t.sum(dim=1) / mask.sum(dim=1)[..., None]


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


# sampling helpers


def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


# discrete vae class


class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1),
        )

    def forward(self, x):
        return self.net(x) + x


class DALLE_Klue_Roberta(nn.Module):
    def __init__(
        self,
        *,
        # dim,
        vae,
        num_text_tokens=10000,
        text_seq_len=256,
        depth,
        heads=8,
        dim_head=64,
        reversible=False,
        attn_dropout=0.0,
        ff_dropout=0,
        sparse_attn=False,
        attn_types=None,
        loss_img_weight=7,
        stable=False,
        sandwich_norm=False,
        shift_tokens=True,
        rotary_emb=False,
        wte_dir=None,
        wpe_dir=None,
    ):
        super().__init__()
        assert isinstance(
            vae, (DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE)
        ), "vae must be an instance of DiscreteVAE"
        image_size = vae.image_size
        num_image_tokens = vae.num_tokens
        image_fmap_size = vae.image_size // (2 ** vae.num_layers)
        image_seq_len = image_fmap_size ** 2

        num_text_tokens = (
            num_text_tokens + text_seq_len
        )  # reserve unique padding tokens for each position (text seq len)

        self.text_emb = torch.load(wte_dir)
        dim = self.text_emb.weight.shape[1]
        self.image_emb = nn.Embedding(num_image_tokens, dim)
        print(dim, image_fmap_size, image_fmap_size)
        self.text_pos_emb = (
            torch.load(wpe_dir) if not rotary_emb else always(0)
        )  # +1 for <bos>
        self.image_pos_emb = (
            AxialPositionalEmbedding(
                dim, axial_shape=(image_fmap_size, image_fmap_size)
            )
            if not rotary_emb
            else always(0)
        )

        self.num_text_tokens = num_text_tokens  # for offsetting logits index and calculating cross entropy loss
        self.num_image_tokens = num_image_tokens

        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len

        seq_len = text_seq_len + image_seq_len
        total_tokens = num_text_tokens + num_image_tokens
        self.total_tokens = total_tokens
        self.total_seq_len = seq_len

        self.vae = vae
        set_requires_grad(self.vae, False)  # freeze VAE from being trained

        self.transformer = Transformer(
            dim=dim,
            causal=True,
            seq_len=seq_len,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            reversible=reversible,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            attn_types=attn_types,
            image_fmap_size=image_fmap_size,
            sparse_attn=sparse_attn,
            stable=stable,
            sandwich_norm=sandwich_norm,
            shift_tokens=shift_tokens,
            rotary_emb=rotary_emb,
        )

        self.stable = stable

        if stable:
            self.norm_by_max = DivideMax(dim=-1)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, self.total_tokens),
        )

        seq_range = torch.arange(seq_len)
        logits_range = torch.arange(total_tokens)

        seq_range = rearrange(seq_range, "n -> () n ()")
        logits_range = rearrange(logits_range, "d -> () () d")

        logits_mask = (
            (seq_range >= text_seq_len) & (logits_range < num_text_tokens)
        ) | ((seq_range < text_seq_len) & (logits_range >= num_text_tokens))

        self.register_buffer("logits_mask", logits_mask, persistent=False)
        self.loss_img_weight = loss_img_weight

    @torch.no_grad()
    @eval_decorator
    def generate_texts(
        self, tokenizer, text=None, *, filter_thres=0.5, temperature=1.0
    ):
        text_seq_len = self.text_seq_len
        if text is None or text == "":
            text_tokens = torch.tensor([[0]]).cuda()
        else:
            text_tokens = (
                torch.tensor(tokenizer.tokenizer.encode(text)).cuda().unsqueeze(0)
            )

        for _ in range(text_tokens.shape[1], text_seq_len):
            device = text_tokens.device

            tokens = self.text_emb(text_tokens)
            tokens += self.text_pos_emb(
                torch.arange(text_tokens.shape[1], device=device)
            )

            seq_len = tokens.shape[1]

            output_transf = self.transformer(tokens)

            if self.stable:
                output_transf = self.norm_by_max(output_transf)

            logits = self.to_logits(output_transf)

            # mask logits to make sure text predicts text (except last token), and image predicts image

            logits_mask = self.logits_mask[:, :seq_len]
            max_neg_value = -torch.finfo(logits.dtype).max
            logits.masked_fill_(logits_mask, max_neg_value)
            logits = logits[:, -1, :]

            filtered_logits = top_k(logits, thres=filter_thres)
            probs = stable_softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            text_tokens = torch.cat((text_tokens, sample), dim=-1)

        padding_tokens = set(
            np.arange(self.text_seq_len) + (self.num_text_tokens - self.text_seq_len)
        )
        texts = [
            tokenizer.tokenizer.decode(text_token, pad_tokens=padding_tokens)
            for text_token in text_tokens
        ]
        return text_tokens, texts

    @torch.no_grad()
    @eval_decorator
    def generate_images(
        self,
        text,
        *,
        clip=None,
        mask=None,
        filter_thres=0.5,
        temperature=1.0,
        img=None,
        num_init_img_tokens=None,
    ):
        vae, text_seq_len, image_seq_len, num_text_tokens = (
            self.vae,
            self.text_seq_len,
            self.image_seq_len,
            self.num_text_tokens,
        )
        total_len = text_seq_len + image_seq_len

        text = text[:, :text_seq_len]  # make sure text is within bounds
        out = text

        if exists(img):
            image_size = vae.image_size
            assert (
                img.shape[1] == 3
                and img.shape[2] == image_size
                and img.shape[3] == image_size
            ), f"input image must have the correct image size {image_size}"

            indices = vae.get_codebook_indices(img)
            num_img_tokens = default(
                num_init_img_tokens, int(0.4375 * image_seq_len)
            )  # OpenAI used 14 * 32 initial tokens to prime
            assert (
                num_img_tokens < image_seq_len
            ), "number of initial image tokens for priming must be less than the total image token sequence length"

            indices = indices[:, :num_img_tokens]
            out = torch.cat((out, indices), dim=-1)

        for cur_len in range(out.shape[1], total_len):
            is_image = cur_len >= text_seq_len

            text, image = out[:, :text_seq_len], out[:, text_seq_len:]

            logits = self(text, image, mask=mask)[:, -1, :]

            filtered_logits = top_k(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            sample -= (
                num_text_tokens if is_image else 0
            )  # offset sampled token if it is an image token, since logit space is composed of text and then image tokens
            out = torch.cat((out, sample), dim=-1)

            if out.shape[1] <= text_seq_len:
                mask = F.pad(mask, (0, 1), value=True)

        text_seq = out[:, :text_seq_len]

        img_seq = out[:, -image_seq_len:]
        images = vae.decode(img_seq)

        if exists(clip):
            scores = clip(text_seq, images, return_loss=False)
            return images, scores

        return images

    def forward(self, text, image=None, mask=None, return_loss=False):
        assert (
            text.shape[-1] == self.text_seq_len
        ), f"the length {text.shape[-1]} of the text tokens you passed in does not have the correct length ({self.text_seq_len})"
        device, total_seq_len = text.device, self.total_seq_len

        # make sure padding in text tokens get unique padding token id

        # text_range = torch.arange(self.text_seq_len, device=device) + (
        #    self.num_text_tokens - self.text_seq_len
        # )
        # print(torch.max(text))
        # print(text)
        # torch.save(text,'text.pt')
        # text = torch.where(text == 3, text_range, text)

        # add <bos>
        # print(text.shape)
        # print(text)
        # print(torch.max(text))
        text = F.pad(text, (1, 0), value=0)
        # print(text.shape)
        # print(text)
        # print(torch.max(text))
        tokens = self.text_emb(text)
        tokens += self.text_pos_emb(torch.arange(text.shape[1], device=device))

        seq_len = tokens.shape[1]

        if exists(image) and not is_empty(image):
            is_raw_image = len(image.shape) == 4

            if is_raw_image:
                image_size = self.vae.image_size
                assert tuple(image.shape[1:]) == (
                    3,
                    image_size,
                    image_size,
                ), f"invalid image of dimensions {image.shape} passed in during training"

                image = self.vae.get_codebook_indices(image)
            image_len = image.shape[1]
            image_emb = self.image_emb(image)
            image_emb += self.image_pos_emb(image_emb)

            tokens = torch.cat((tokens, image_emb), dim=1)

            seq_len += image_len

        # when training, if the length exceeds the total text + image length
        # remove the last token, since it needs not to be trained

        if tokens.shape[1] > total_seq_len:
            seq_len -= 1
            tokens = tokens[:, :-1]

        if self.stable:
            alpha = 0.1
            tokens = tokens * alpha + tokens.detach() * (1 - alpha)

        out = self.transformer(tokens)

        if self.stable:
            out = self.norm_by_max(out)

        logits = self.to_logits(out)

        # mask logits to make sure text predicts text (except last token), and image predicts image

        logits_mask = self.logits_mask[:, :seq_len]
        max_neg_value = -torch.finfo(logits.dtype).max
        logits.masked_fill_(logits_mask, max_neg_value)

        if not return_loss:
            return logits

        assert exists(image), "when training, image must be supplied"

        offsetted_image = image + self.num_text_tokens
        labels = torch.cat((text[:, 1:], offsetted_image), dim=1)

        logits = rearrange(logits, "b n c -> b c n")

        loss_text = F.cross_entropy(
            logits[:, :, : self.text_seq_len], labels[:, : self.text_seq_len]
        )
        loss_img = F.cross_entropy(
            logits[:, :, self.text_seq_len :], labels[:, self.text_seq_len :]
        )

        loss = (loss_text + self.loss_img_weight * loss_img) / (
            self.loss_img_weight + 1
        )
        return loss


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
                save_model(f"{args.save_path}/dalle_uk.pt")
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
    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # Configuration
    VAE_CFG = vae_config()

    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")  # Korean Tokenizer
    DALLE_CFG = dalle_config(tokenizer)

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
    if args.transformer == "basic":
        initial_transformation = basic_transformer(VAE_CFG)
    # elif args.transformer == 'center_crop' :
    #     ...

    dataset_visual = ImgDatasetExample(
        image_folder=args.image_folder, image_transform=initial_transformation
    )

    dataloader_visual = DataLoader(
        dataset=dataset_visual, batch_size=args.batch_size, shuffle=True
    )

    # images = next(iter(dataloader_visual))
    # show_transform(make_grid(images, nrow=6), title= vars(initial_transformation))

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
