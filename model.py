import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import repeat
from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange

from dalle_pytorch import DiscreteVAE
from dalle_pytorch.vae import OpenAIDiscreteVAE, VQGanVAE

from dalle_pytorch.transformer import Transformer, DivideMax
from utils import *
from tqdm import tqdm

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
            nn.LayerNorm(dim),
            nn.Linear(dim, self.total_tokens),
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
    def generate_images(
        self,
        encoded_text,
        *,
        clip=None,
        filter_thres=0.5,
        temperature=1.0,
        img=None,
        num_init_img_tokens=None,
        img_num=1,
    ):
        text = encoded_text['input_ids']
        text=repeat(text,'() n -> b n',b=img_num)
        mask=encoded_text['attention_mask']
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

        for cur_len in tqdm(range(out.shape[1], total_len)):
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



        img_seq = out[:, -image_seq_len:]
        images = vae.decode(img_seq)

        if exists(clip):
            #encoded_text = encoded_text.to("cuda")
            text_embeds, image_embeds = clip(encoded_text, images)
            logits = text_embeds @ image_embeds.T
            return images, logits

        return images

    def forward(self, text, image=None, mask=None, return_loss=False):
        assert (
            text.shape[-1] == self.text_seq_len
        ), f"the length {text.shape[-1]} of the text tokens you passed in does not have the correct length ({self.text_seq_len})"
        device, total_seq_len = text.device, self.total_seq_len

        # make sure padding in text tokens get unique padding token id
        text = F.pad(text, (1, 0), value=0)

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
