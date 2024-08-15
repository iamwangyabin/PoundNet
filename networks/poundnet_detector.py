import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F

from networks.clip import clip
from networks.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.model.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'IVLP',
                      "vision_depth": cfg.model.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.model.PROMPT_DEPTH_TEXT, "vision_ctx": cfg.model.N_CTX_VISION,
                      "language_ctx": cfg.model.N_CTX_TEXT}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        if len(prompts.shape) == 4:
            prompts = torch.flatten(prompts, start_dim=0, end_dim=1)
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class ARPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        n_cls = len(classnames)
        assert cfg.model.PROMPT_DEPTH_TEXT >= 1, ("In Independent VL prompting, Language prompt depth should be >=1 \n "
                                                  "Please use VPT trainer if you want to learn only vision branch  ")
        n_ctx = cfg.model.N_CTX_TEXT
        ctx_init = cfg.model.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        self.prompt_num = cfg.model.PROMPT_NUM_TEXT

        # only reserve random initialization, remove ctx_init
        ctx_vectors = torch.empty(self.prompt_num*2, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)
        # self.ctx = nn.Parameter(ctx_vectors)

        ctx_positive = ctx_vectors[:self.prompt_num, :, :]
        ctx_negative = ctx_vectors[self.prompt_num:, :, :]
        self.ctx_positive = nn.Parameter(ctx_positive)
        self.ctx_negative = nn.Parameter(ctx_negative)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        positive_prompts = [prompt_prefix + " " +  name   for name in classnames]
        negative_prompts = [prompt_prefix + " " + name  for name in classnames]
        positive_tokenized_prompts = torch.cat([clip.tokenize(p) for p in positive_prompts])
        negative_tokenized_prompts = torch.cat([clip.tokenize(p) for p in negative_prompts])
        with torch.no_grad():
            positive_embedding = clip_model.token_embedding(positive_tokenized_prompts).type(dtype)
            negative_embedding = clip_model.token_embedding(negative_tokenized_prompts).type(dtype)

        positive_embedding = positive_embedding.view(positive_embedding.shape[0], 1, positive_embedding.shape[1], positive_embedding.shape[2])
        negative_embedding = negative_embedding.view(negative_embedding.shape[0], 1, negative_embedding.shape[1], negative_embedding.shape[2])
        positive_embedding = positive_embedding.repeat(1, self.prompt_num, 1, 1)
        negative_embedding = negative_embedding.repeat(1, self.prompt_num, 1, 1)
        embedding = torch.cat([positive_embedding, negative_embedding], dim=1)
        positive_tokenized_prompts = positive_tokenized_prompts.view(positive_tokenized_prompts.shape[0], 1, positive_tokenized_prompts.shape[1])
        negative_tokenized_prompts = negative_tokenized_prompts.view(negative_tokenized_prompts.shape[0], 1, negative_tokenized_prompts.shape[1])
        positive_tokenized_prompts = positive_tokenized_prompts.repeat(1, self.prompt_num, 1)
        negative_tokenized_prompts = negative_tokenized_prompts.repeat(1, self.prompt_num, 1)
        tokenized_prompts = torch.cat([positive_tokenized_prompts, negative_tokenized_prompts], dim=1)
        # tokenized_prompts = tokenized_prompts.view(tokenized_prompts.shape[0]*tokenized_prompts.shape[1], -1)
        tokenized_prompts = tokenized_prompts.permute(1, 0, 2).contiguous().view(tokenized_prompts.shape[0]*tokenized_prompts.shape[1], -1)
        # import pdb;pdb.set_trace()
        self.register_buffer("token_prefix", embedding[:, :, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, :, 1 + n_ctx:, :])  # positive prompt CLS, EOS
        self.register_buffer("positive_token_prefix", embedding[:, :self.prompt_num, :1, :])  # SOS
        self.register_buffer("positive_token_suffix", embedding[:, :self.prompt_num, 1 + n_ctx:, :])  # positive prompt CLS, EOS
        self.register_buffer("negative_token_prefix", embedding[:, self.prompt_num:, :1, :])  # SOS
        self.register_buffer("negative_token_suffix", embedding[:, self.prompt_num:, 1 + n_ctx:, :])
        self.positive_tokenized_prompts = positive_tokenized_prompts.view(
            positive_tokenized_prompts.shape[0] * positive_tokenized_prompts.shape[1], -1)
        self.negative_tokenized_prompts = negative_tokenized_prompts.view(
            negative_tokenized_prompts.shape[0] * negative_tokenized_prompts.shape[1], -1)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens


    def forward(self):
        ctx_positive = self.ctx_positive
        ctx_negative = self.ctx_negative
        if ctx_negative.shape[0] == 0:
            if ctx_positive.dim() == 3:
                ctx = ctx_positive.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            else:
                ctx = ctx_positive
        else:
            if ctx_positive.dim() == 3:
                diff = ctx_positive.shape[1] - ctx_negative.shape[1]
                additional_rows = torch.zeros((ctx_negative.shape[0], diff, ctx_negative.shape[2])).cuda()
                additional_rows = additional_rows.to(ctx_negative.dtype)
                ctx_negative = torch.cat([additional_rows, ctx_negative], dim=1)
                ctx = torch.cat([ctx_positive, ctx_negative], dim=0)
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            else:
                ctx = torch.cat([ctx_positive, ctx_negative], dim=1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls,1+n_neg, 1, dim)
                ctx,  # (n_cls,1+n_neg, n_ctx, dim)
                suffix,  # (n_cls,1+n_neg, *, dim)
            ],
            dim=2,
        )

        new_order = [item for i in range(self.prompt_num) for item in (self.prompt_num + i, i)] #[2, 0, 3, 1]
        prompts = torch.cat([prompts[:, i, :, :] for i in new_order], dim=0)
        return prompts

    def forward_general_realfake(self):
        ctx_positive = self.ctx_positive
        ctx_negative = self.ctx_negative
        if ctx_negative.shape[0] == 0:
            if ctx_positive.dim() == 3:
                ctx = ctx_positive.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            else:
                ctx = ctx_positive
        else:
            if ctx_positive.dim() == 3:
                diff = ctx_positive.shape[1] - ctx_negative.shape[1]
                additional_rows = torch.zeros((ctx_negative.shape[0], diff, ctx_negative.shape[2])).cuda()
                additional_rows = additional_rows.to(ctx_negative.dtype)
                ctx_negative = torch.cat([additional_rows, ctx_negative], dim=1)
                ctx = torch.cat([ctx_positive, ctx_negative], dim=0)
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            else:
                ctx = torch.cat([ctx_positive, ctx_negative], dim=1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls,1+n_neg, 1, dim)
                ctx,  # (n_cls,1+n_neg, n_ctx, dim)
                suffix,  # (n_cls,1+n_neg, *, dim)
            ],
            dim=2,
        )
        fakep = prompts.mean(0)[:self.prompt_num].mean(0)
        realp = prompts.mean(0)[self.prompt_num:].mean(0)
        prompts = torch.stack([realp, fakep],dim=0)

        return prompts


class PoundNet(nn.Module):

    def __init__(self, cfg):
        super(PoundNet, self).__init__()

        self.cfg = cfg
        print(f"Loading CLIP (backbone: {cfg.model.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        classnames = cfg.datasets.train.multicalss_names
        # import pdb;pdb.set_trace()
        self.prompt_learner = ARPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        for name, param in self.named_parameters():
            if "ctx" in name or "VPT" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

    def forward(self, image):
        prompts = self.prompt_learner.forward_general_realfake()
        column_maxes = self.tokenized_prompts.max(dim=0)[0] + torch.arange(self.tokenized_prompts.size(1))
        new_token = column_maxes.repeat(prompts.size(0), 1)
        text_features = self.text_encoder(prompts, new_token)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = self.logit_scale.exp() * image_features @ text_features.t()
        return {'logits': logits, 'features': image_features}





