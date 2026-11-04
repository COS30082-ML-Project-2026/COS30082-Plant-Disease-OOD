"""Contract tests without downloading weights or running official test images."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from approach3_clip_improvements.approaches.clip_improvement_core import (
    configure_visual, extract_state, interpolate, prompt_groups, strict_load, write_json,
)
from approach3_clip_improvements.approaches.clip_improvement_training import training_step
from approach3_clip_improvements.approaches.clip_improvements import ImprovementRun, default_hp
from common.dataset import get_crop_and_disease_names, get_pairs


class TinyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(3, 4)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList(nn.Linear(4, 4) for _ in range(12))
        self.proj = nn.Linear(4, 4)

    def forward(self, images):
        features = self.input(images.mean((-1, -2)))
        for block in self.transformer.resblocks:
            features = features + 0.1 * block(features).tanh()
        return self.proj(features)


class TinyClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = TinyVisual()
        self.token_embedding = nn.Embedding(400, 4)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def encode_image(self, images):
        return self.visual(images)


class ImprovementContracts(unittest.TestCase):
    def test_interpolation_endpoints_and_nonfloating_buffer(self):
        base = dict(weight=torch.tensor([1., 2.]), counter=torch.tensor(4))
        tuned = dict(weight=torch.tensor([5., 8.]), counter=torch.tensor(9))
        self.assertTrue(torch.equal(interpolate(base, tuned, 0)['weight'], base['weight']))
        self.assertTrue(torch.equal(interpolate(base, tuned, 1)['weight'], tuned['weight']))
        self.assertEqual(interpolate(base, tuned, 0.5)['counter'].item(), 4)
        self.assertTrue(torch.equal(interpolate(base, tuned, 0.5)['weight'], torch.tensor([3., 5.])))
        with self.assertRaises(ValueError):
            interpolate(base, tuned, 1.1)

    def test_strict_checkpoint_and_wrapper_rejection(self):
        model = TinyClip()
        state = model.state_dict()
        wrapped = {'state_dict': {'module.'+k: v for k,v in state.items()}}
        self.assertEqual(strict_load(model, extract_state(wrapped), 'test')['missing'], [])
        broken = dict(state)
        broken['logit_scale'] = torch.ones(2)
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            strict_load(model, broken, 'test')
        broken = {k:v.clone() for k,v in state.items()}
        broken['logit_scale'].fill_(float('nan'))
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            strict_load(model, broken, 'test')

    def test_exact_partial_freeze_and_teacher_gradients(self):
        for variant in ('partial6', 'visual_full'):
            model = TinyClip()
            teacher = copy.deepcopy(model).eval().requires_grad_(False)
            names = configure_visual(model, variant)
            self.assertTrue(all(n.startswith('visual.') for n in names))
            self.assertFalse(model.token_embedding.weight.requires_grad)
            self.assertFalse(model.logit_scale.requires_grad)
            engine = SimpleNamespace(model=model, teacher=teacher, device=torch.device('cpu'), hp={'precision':'float32'})
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            text = F.normalize(torch.randn(333, 4), dim=1)
            before = model.visual.proj.weight.detach().clone() if variant == 'visual_full' else model.visual.transformer.resblocks[-1].weight.detach().clone()
            result = training_step(engine, torch.randn(2,3,4,4), torch.tensor([0,1]), optimizer, text, 0.1, names)
            after = model.visual.proj.weight if variant == 'visual_full' else model.visual.transformer.resblocks[-1].weight
            self.assertFalse(torch.equal(before, after))
            self.assertTrue(torch.isfinite(torch.tensor(result['loss'])))
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_healthy_prompts_use_real_mapping(self):
        pairs = get_pairs()
        _, diseases = get_crop_and_disease_names()
        groups = prompt_groups(pairs, default_hp(), 'ensemble3')
        self.assertEqual(len(groups), 333)
        for (_, disease), group in zip(pairs.tolist(), groups):
            if diseases[disease].casefold() == 'healthy':
                self.assertTrue(all('healthy' in text and 'with healthy' not in text for text in group))

    def test_selection_gate_and_output_collision(self):
        run = object.__new__(ImprovementRun)
        run.phase = 'selection'
        with self.assertRaisesRegex(ValueError, 'prohibited'):
            run._evaluate('forbidden', {}, 'single', {})
        run.selections = {}
        with self.assertRaisesRegex(ValueError, 'All five'):
            run.seal_selection()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'result.json'
            write_json(target, {'value': 1})
            with self.assertRaises(FileExistsError):
                write_json(target, {'value': 2})

    def test_debug_checkpoint_roundtrip_is_ephemeral(self):
        run = object.__new__(ImprovementRun)
        run.model, run.pairs, run.hp = TinyClip(), get_pairs(), default_hp()
        run.run_id, run.debug, run.debug_states = 'unit', True, {}
        with tempfile.TemporaryDirectory() as directory:
            run.checkpoint_dir = Path(directory)
            state = run.model.state_dict()
            path = run._save_state('unit_roundtrip', state, {'validation_only': True}, 'single')
            self.assertFalse(Path(path).exists())
            self.assertEqual(run.debug_states['unit_roundtrip'].keys(), state.keys())

    def test_full_baseline_reencodes_its_own_text_encoder(self):
        run = object.__new__(ImprovementRun)
        run.model, run.phase, run.device = TinyClip(), 'frozen', torch.device('cpu')
        run.base = {k:v.clone() for k,v in run.model.state_dict().items()}
        tuned = {k:v.clone() for k,v in run.base.items()}
        tuned['token_embedding.weight'] += 0.1
        run.pairs, run.hp, run.debug, run.counts = get_pairs(), default_hp(), True, {'seen':1,'unseen':1}
        run.loaders = lambda: (None, None, None, None)
        run.tokenizer = lambda texts: torch.ones(len(texts), 2, dtype=torch.long)
        run.text = lambda style: self.fail('Full baseline reused original frozen text')
        with tempfile.TemporaryDirectory() as directory:
            run.result_dir = Path(directory)
            with patch('approach3_clip_improvements.approaches.clip_improvements.encode_groups', return_value=torch.randn(333,4)) as encode:
                with patch('approach3_clip_improvements.approaches.clip_improvements.evaluation_loader', side_effect=lambda x:x):
                    with patch('approach3_clip_improvements.approaches.clip_improvements.evaluate', return_value={}):
                        run._evaluate('full', tuned, 'single', {})
            self.assertIs(encode.call_args.args[0], run.model)


if __name__ == '__main__':
    unittest.main()
