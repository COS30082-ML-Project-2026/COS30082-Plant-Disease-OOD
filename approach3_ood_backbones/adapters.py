"""Backbone-specific inference; weights are never downloaded without opt-in."""
import hashlib
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import torch
import torch.nn.functional as F


def versions():
    result = {}
    for name in ('torch', 'torchvision', 'open_clip_torch', 'transformers', 'huggingface_hub'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def snapshot(repository, revision, allow_download, patterns):
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(repository, revision=revision, allow_patterns=patterns,
                                  local_files_only=not allow_download))
    return path, path.name


def weight_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class OpenClipAdapter:
    def __init__(self, spec, device, allow_download=False):
        import open_clip
        self.device = torch.device(device)
        self.spec = spec
        if spec['backend'] == 'open_clip_hub':
            path, revision = snapshot(spec['repository'], spec['revision'], allow_download,
                                      ['open_clip_config.json', 'open_clip_pytorch_model.bin',
                                       'open_clip_model.safetensors', '*token*', '*.txt'])
            import json
            config = json.loads((path / 'open_clip_config.json').read_text())
            weight = next((path / p for p in ('open_clip_model.safetensors', 'open_clip_pytorch_model.bin')
                           if (path / p).is_file()), None)
            if weight is None:
                raise FileNotFoundError('BioCLIP checkpoint not cached; explicitly allow download first.')
            # Register its exact published architecture and preprocessing locally.
            # add_model_config expects a model_cfg directly, not the Hub wrapper.
            model = open_clip.model.CLIP(**config['model_cfg'])
            open_clip.load_checkpoint(model, str(weight))
            from open_clip.transform import PreprocessCfg, image_transform_v2
            preprocess_cfg = dict(config.get('preprocess_cfg', {}))
            preprocess_cfg.setdefault('size', model.visual.image_size)
            self.preprocess = image_transform_v2(PreprocessCfg(**preprocess_cfg), is_train=False)
            self.tokenizer = open_clip.get_tokenizer('ViT-B-16')
            self.metadata = dict(resolved_revision=revision, weight_sha256=weight_hash(weight),
                                 tokenizer='OpenCLIP ViT-B-16 SimpleTokenizer (official BioCLIP convention)')
        else:
            from urllib.parse import urlparse
            architecture = spec['architecture']
            info = open_clip.get_pretrained_cfg(architecture, spec['pretrained'])
            # OpenAI published JIT checkpoint has a URL-embedded SHA256.
            url = info.get('url', '')
            if not url:
                raise ValueError('Expected official OpenAI checkpoint URL.')
            cache = Path.home() / '.cache' / 'clip'
            weight = cache / Path(urlparse(url).path).name
            expected = url.rstrip('/').split('/')[-2]
            source_revision = expected
            source_kind = 'OpenAI JIT'
            if not weight.is_file() and not allow_download and info.get('hf_hub'):
                # Reuse previously cached OpenCLIP conversion without network access.
                try:
                    cached, source_revision = snapshot(info['hf_hub'].rstrip('/'), 'main', False,
                                                       ['open_clip_pytorch_model.bin', 'open_clip_model.safetensors'])
                    weight = next(p for p in [cached / 'open_clip_model.safetensors', cached / 'open_clip_pytorch_model.bin'] if p.is_file())
                    source_kind = 'OpenCLIP Hugging Face conversion of OpenAI weights'
                except (OSError, StopIteration):
                    pass
            if not weight.is_file():
                if not allow_download:
                    raise FileNotFoundError(f'Missing {weight}; use --allow-download only after reviewing model sizes.')
                cache.mkdir(parents=True, exist_ok=True)
                from open_clip.pretrained import download_pretrained
                weight = Path(download_pretrained(info, prefer_hf_hub=False, cache_dir=str(cache)))
            digest = weight_hash(weight)
            if source_kind == 'OpenAI JIT' and len(expected) == 64 and digest != expected:
                raise ValueError('OpenAI checkpoint SHA256 mismatch.')
            model, _, self.preprocess = open_clip.create_model_and_transforms(
                architecture, pretrained=str(weight), force_quick_gelu=True)
            self.tokenizer = open_clip.get_tokenizer(architecture)
            self.metadata = dict(checkpoint_url=url, weight_sha256=digest,
                                 resolved_revision=source_revision, source_kind=source_kind)
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.metadata.update(spec=spec, preprocessing=repr(self.preprocess), versions=versions(),
                             parameters=sum(p.numel() for p in model.parameters()),
                             scoring='exp(logit_scale) * normalized image @ normalized text.T + optional bias')

    @torch.no_grad()
    def encode_images(self, images):
        pixels = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        return F.normalize(self.model.encode_image(pixels).float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, texts):
        return F.normalize(self.model.encode_text(self.tokenizer(texts).to(self.device)).float(), dim=-1)

    def similarity(self, images, text):
        scores = self.model.logit_scale.exp().float() * images @ text.T
        bias = getattr(self.model, 'logit_bias', None)
        return scores if bias is None else scores + bias.float()


class Siglip2Adapter:
    def __init__(self, spec, device, allow_download=False):
        from transformers import AutoModel, AutoProcessor
        self.device = torch.device(device)
        path, revision = snapshot(spec['repository'], spec['revision'], allow_download,
                                  ['*.json', '*.model', '*.safetensors', '*.txt'])
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).to(self.device).eval().requires_grad_(False)
        self.metadata = dict(spec=spec, resolved_revision=revision, versions=versions(),
                             preprocessing=self.processor.image_processor.to_dict(),
                             tokenizer=dict(padding='max_length', max_length=64, truncation=True, lowercase=True),
                             parameters=sum(p.numel() for p in self.model.parameters()),
                             scoring='exp(logit_scale) * normalized image @ normalized text.T + logit_bias; raw logits')

    @torch.no_grad()
    def encode_images(self, images):
        inputs = self.processor(images=images, return_tensors='pt').to(self.device)
        result = self.model.get_image_features(**inputs)
        if not torch.is_tensor(result):
            raise TypeError('SigLIP2 feature API changed; use pinned Transformers 4.52.2.')
        return F.normalize(result.float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, texts):
        inputs = self.processor(text=[text.lower() for text in texts], padding='max_length',
                                max_length=64, truncation=True, return_tensors='pt').to(self.device)
        result = self.model.get_text_features(**inputs)
        if not torch.is_tensor(result):
            raise TypeError('SigLIP2 feature API changed; use pinned Transformers 4.52.2.')
        return F.normalize(result.float(), dim=-1)

    def similarity(self, images, text):
        return self.model.logit_scale.exp().float() * images @ text.T + self.model.logit_bias.float()


def load_adapter(spec, device, allow_download=False):
    backend = spec['backend']
    if backend == 'transformers':
        return Siglip2Adapter(spec, device, allow_download)
    if backend in ('open_clip', 'open_clip_hub'):
        return OpenClipAdapter(spec, device, allow_download)
    raise ValueError(f'Unsupported adapter backend: {backend}')
