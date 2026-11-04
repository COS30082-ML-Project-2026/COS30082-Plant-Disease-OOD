"""Five CLIP improvement experiments with a sealed validation-only selection phase."""
import copy
import gc
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import open_clip
import pandas as pd
import torch

from common import config as cfg
from common.evaluate import evaluate
from approaches.clip_improvement_core import (
    check_data, checked_loaders, encode_groups, evaluation_loader, extract_state,
    file_hash, interpolate, make_predict, prompt_groups, read_checkpoint, require,
    resolve_partial_path, seed_all, strict_load, train_manifest, write_json,
)
from approaches.clip_improvement_training import fit_candidate, validation


@contextmanager
def evaluation_directory(path):
    previous = cfg.RESULTS_DIR
    cfg.RESULTS_DIR = path
    try:
        yield
    finally:
        cfg.RESULTS_DIR = previous


def default_hp():
    return dict(backbone='ViT-B-16', pretrained='openai', seed=42, lr=1e-5, weight_decay=0.05,
                epochs=10, batch_size=8, optimizer='AdamW', precision='float32',
                gradient_checkpointing=True, norm='clip', robust=True, workers=cfg.NUM_WORKERS,
                prompt='a photo of a {crop} leaf with {disease}', debug_samples=2,
                debug_batch_size=1, debug_workers=0, text_chunk_size=16,
                alpha_grid=[0.0, 0.25, 0.5, 0.75, 1.0], lambda_grid=[0.1, 1.0],
                prompt_styles=['ensemble3', 'ensemble5'],
                healthy_templates=['a photo of a healthy {crop} leaf', 'a close-up photo of a healthy {crop} leaf'],
                ensemble_templates=['a photo of a {crop} leaf with {disease}',
                                    'a close-up photo of a {crop} leaf affected by {disease}',
                                    'a photo of {disease} on a {crop} leaf',
                                    'a plant pathology image of {crop} with {disease}',
                                    'a {crop} leaf showing symptoms of {disease}'])


class ImprovementRun:
    def __init__(self, hp, debug=True, partial_path=None, baseline_results=None):
        self.requested_hp, self.hp, self.debug = copy.deepcopy(hp), copy.deepcopy(hp), debug
        if debug:
            self.hp.update(epochs=1, batch_size=hp['debug_batch_size'], workers=hp['debug_workers'],
                           alpha_grid=[0.0, 0.5, 1.0], lambda_grid=[hp['lambda_grid'][0]])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.partial_path = resolve_partial_path(partial_path or os.getenv('CLIP_PARTIAL6'))
        self.baseline_results = baseline_results or os.getenv('CLIP_BASELINE_RESULTS')
        self.pairs, self.selections, self.text_bank, self.debug_states = check_data(), {}, {}, {}
        self.original_path = cfg.ROOT_DIR / 'approaches' / 'approach3_clip.ipynb'
        self.original_hash = file_hash(self.original_path)
        self._check_settings()
        self._create_run()
        self._setup_models()
        self._inspect_splits()
        write_json(self.result_dir / 'setup.json', self.metadata())

    def _check_settings(self):
        require(self.hp['seed'] == cfg.SEED == 42, 'Use shared seed 42 for a comparable split.')
        require(self.hp['precision'] in ('float32', 'bf16'), 'precision must be float32 or bf16.')
        require(all(0 <= a <= 1 for a in self.hp['alpha_grid']), 'Invalid WiSE alpha grid.')
        require(all(x > 0 for x in self.hp['lambda_grid']), 'Preservation lambda grid must be positive.')
        require(self.hp['optimizer'] == 'AdamW', 'Use the existing AdamW convention.')
        require(self.hp['epochs'] > 0 and self.hp['batch_size'] > 0, 'Epochs/batch size must be positive.')
        require(set(self.hp['prompt_styles']) <= {'ensemble3', 'ensemble5'}, 'Unknown prompt ensemble.')
        if self.hp['precision'] == 'bf16':
            require(self.device.type == 'cuda' and torch.cuda.is_bf16_supported(), 'bf16 CUDA not available.')

    def _create_run(self):
        minimum = (768 if self.debug else 4096) * 1024**2
        require(shutil.disk_usage(cfg.ROOT_DIR).free >= minimum,
                f'Need at least {minimum // 1024**2} MB free for this mode. Free disk space; no outputs were replaced.')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.run_id = ('debug_' if self.debug else 'full_') + stamp + '_' + uuid4().hex[:8]
        self.result_dir = cfg.ROOT_DIR / 'results' / 'clip_improvements' / self.run_id
        self.checkpoint_dir = cfg.CHECKPOINT_DIR / 'clip_improvements' / self.run_id
        self.result_dir.mkdir(parents=True, exist_ok=False)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=False)
        self.phase = 'selection'
        print('New run:', self.run_id, '\nResults:', self.result_dir, '\nWeights:', self.checkpoint_dir)

    def _setup_models(self):
        seed_all(self.hp['seed'])
        self.teacher, _, transform = open_clip.create_model_and_transforms(
            self.hp['backbone'], pretrained=self.hp['pretrained'], force_quick_gelu=True)
        self.teacher.cpu().eval().requires_grad_(False)
        self.base = self.teacher.state_dict()
        require(tuple(transform.transforms[-1].mean) == cfg.NORM_STATS['clip'][0], 'CLIP normalization mismatch.')
        self.model = open_clip.create_model(self.hp['backbone'], force_quick_gelu=True).to(self.device)
        self.model.requires_grad_(False)
        self.model.set_grad_checkpointing(self.hp['gradient_checkpointing'])
        self.tokenizer = open_clip.get_tokenizer(self.hp['backbone'])
        checkpoint = read_checkpoint(self.partial_path)
        self.partial = extract_state(checkpoint)
        self.compatibility = strict_load(self.model, self.partial, 'Existing Partial-6')
        self._check_partial_metadata(checkpoint)
        self.source_hash = file_hash(self.partial_path)

    def _check_partial_metadata(self, checkpoint):
        require(checkpoint.get('model_name') == 'approach3_clip_6', 'Expected the formal Partial-6 checkpoint.')
        require(torch.equal(checkpoint.get('pairs', torch.empty(0)), self.pairs), 'Checkpoint candidate order differs.')
        expected_prompts = [g[0] for g in prompt_groups(self.pairs, self.hp, 'single')]
        require(checkpoint.get('prompts') == expected_prompts, 'Checkpoint prompt/name mapping differs.')
        require(not checkpoint.get('environment', {}).get('debug', True), 'Use a formal, non-DEBUG Partial-6.')
        for key in ('backbone', 'pretrained', 'seed', 'lr', 'weight_decay', 'epochs', 'batch_size', 'optimizer', 'prompt'):
            require(checkpoint['hp'].get(key) == self.requested_hp[key],
                    f'Control {key} differs from Partial-6; document a separate experiment instead.')
        require(all(torch.equal(self.partial[k], v) for k, v in self.base.items() if not k.startswith('visual.')),
                'Partial-6 text/logit weights differ from original CLIP; frozen-text cache would be unsafe.')

    def loaders(self):
        previous = cfg.NUM_WORKERS
        cfg.NUM_WORKERS = self.hp['workers']
        try:
            loaders, signatures = checked_loaders(self.hp, self.debug, self.hp['workers'])
        finally:
            cfg.NUM_WORKERS = previous
        if hasattr(self, 'split_signatures'):
            require(signatures == self.split_signatures, 'Shared split membership changed; stop this run.')
        else:
            self.split_signatures = signatures
        return loaders

    def _inspect_splits(self):
        train, val, seen, unseen = self.loaders()
        self.split = train_manifest(train, val)
        self.counts = dict(train=len(train.dataset), validation=len(val.dataset),
                           seen=len(seen.dataset), unseen=len(unseen.dataset))
        if not self.debug:
            require(self.counts == dict(train=7223, validation=777, seen=2000, unseen=1448),
                    f'Full split differs from the original run: {self.counts}. Check shared CSV/split versions.')
        print('Device:', self.device, 'effective settings:', self.hp, '\nCounts:', self.counts)

    def metadata(self):
        packages = {p: version(p) for p in ('torch', 'torchvision', 'open_clip_torch', 'numpy', 'pandas')}
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=cfg.ROOT_DIR, capture_output=True, text=True)
        return dict(run_id=self.run_id, debug=self.debug, hp=self.hp, requested_hp=self.requested_hp,
                    python=sys.version, packages=packages, device=str(self.device), cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name() if self.device.type == 'cuda' else None,
                    repository_commit=revision.stdout.strip(), split=self.split, samples=self.counts,
                    partial_checkpoint=str(self.partial_path), partial_sha256=self.source_hash,
                    original_notebook_sha256=self.original_hash, compatibility=self.compatibility,
                    selection_metric='seen validation Top-1 over all 333 candidates; ties favour smaller grid value',
                    limitation='No additional pairs held out. Seen validation cannot estimate unseen transfer.',
                    teacher='Original OpenAI CLIP ViT-B/16, frozen on CPU',
                    checkpoint_storage='DEBUG: verified disk round-trip, then kept in CPU memory' if self.debug else 'Permanent full checkpoints',
                    runtime_changes='Evaluation uses zero workers for deterministic perturbations; training split unchanged.')

    def text(self, style):
        if style not in self.text_bank:
            groups = prompt_groups(self.pairs, self.hp, style)
            self.text_bank[style] = encode_groups(self.teacher, self.tokenizer, groups, torch.device('cpu'),
                                                 self.hp['text_chunk_size']).to(self.device)
        return self.text_bank[style]

    def _selecting(self, name):
        require(self.phase == 'selection', 'Selection already sealed. Start a new run to change settings.')
        require(name not in self.selections, f'{name} already ran; refuse to overwrite. Start a new run.')
        require(file_hash(self.original_path) == self.original_hash, 'Original notebook changed during this run.')

    def _save_state(self, name, state, record, style):
        path = self.checkpoint_dir / (name + '.pt')
        require(not path.exists(), f'Checkpoint collision: {path}; use a new run ID.')
        payload = dict(model_name=name, state_dict=state, hp=self.hp, pairs=self.pairs,
                       prompts=prompt_groups(self.pairs, self.hp, style), selection=record,
                       run_id=self.run_id, debug=self.debug, text_frozen=True)
        with path.open('xb') as stream:
            torch.save(payload, stream)
        strict_load(self.model, extract_state(read_checkpoint(path)), name + ' saved reload')
        if self.debug:
            self.debug_states[name] = state
            path.unlink()  # Only this newly generated DEBUG round-trip file; source checkpoints are untouched.
        return str(path)

    def _store(self, name, state, record, style):
        record.update(experiment=name, prompt_style=style, samples=self.counts, split=self.split,
                      validation_only=True, text_frozen=True, checkpoint_ephemeral=self.debug)
        record['checkpoint'] = self._save_state(name, state, record.copy(), style)
        self.selections[name] = record
        write_json(self.result_dir / (name + '_selection.json'), record)
        return record

    def choose_alpha(self, tuned, style):
        _, val, _, _ = self.loaders()
        history = []
        for alpha in self.hp['alpha_grid']:
            strict_load(self.model, interpolate(self.base, tuned, alpha), f'WiSE alpha={alpha}')
            score = validation(self.model, self.text(style), val, self.pairs)
            history.append(dict(alpha=alpha, **score))
            print('WiSE validation:', history[-1], flush=True)
        selected = max(history, key=lambda r: (r['top1'], -r['alpha']))
        return selected['alpha'], history

    def wise_ft(self):
        self._selecting('E1_wise')
        alpha, history = self.choose_alpha(self.partial, 'single')
        record = dict(alpha=alpha, alpha_history=history, retrained=False, trainable_parameters=0)
        return self._store('E1_wise', interpolate(self.base, self.partial, alpha), record, 'single')

    def prompt_ensemble(self):
        self._selecting('E2_prompts')
        strict_load(self.model, self.partial, 'E2 existing Partial-6')
        _, val, _, _ = self.loaders()
        history = [dict(style=style, **validation(self.model, self.text(style), val, self.pairs))
                   for style in self.hp['prompt_styles']]
        chosen = max(history, key=lambda r: (r['top1'], -int(r['style'].removeprefix('ensemble'))))
        record = dict(prompt_history=history, retrained=False, trainable_parameters=0)
        return self._store('E2_prompts', self.partial, record, chosen['style'])

    def preservation(self):
        self._selecting('E3_preserve')
        records, best_record, best_state = [], None, None
        for weight in self.hp['lambda_grid']:
            record, state = fit_candidate(self, 'partial6', weight, 'single', f'E3_lambda_{weight}')
            records.append(record)
            if best_record is None or (record['val_top1'], -weight) > (best_record['val_top1'], -best_record['lambda_preservation']):
                best_record, best_state = record, state
        return self._store('E3_preserve', best_state,
                           dict(best_record, lambda_history=records, retrained=True), 'single')

    def visual_full(self):
        self._selecting('E4_visual_full')
        record, state = fit_candidate(self, 'visual_full', 0.0, 'single', 'E4_visual_full')
        return self._store('E4_visual_full', state, dict(record, retrained=True), 'single')

    def combined(self):
        self._selecting('E5_combined')
        require('E3_preserve' in self.selections and 'E2_prompts' in self.selections, 'Run E2/E3 before E5.')
        weight = self.selections['E3_preserve']['lambda_preservation']
        style = self.selections['E2_prompts']['prompt_style']
        record, state = fit_candidate(self, 'visual_full', weight, style, 'E5_combined_raw')
        raw_path = self._save_state('E5_combined_raw', state, record, style)
        alpha, history = self.choose_alpha(state, style)
        record.update(alpha=alpha, alpha_history=history, raw_checkpoint=raw_path, retrained=True,
                      lambda_source='E3 validation winner, transferred unchanged', prompt_source='E2 validation winner')
        return self._store('E5_combined', interpolate(self.base, state, alpha), record, style)

    def seal_selection(self):
        require(set(self.selections) == {'E1_wise', 'E2_prompts', 'E3_preserve', 'E4_visual_full', 'E5_combined'},
                'All five experiments must finish selection before official evaluation.')
        require(self.phase == 'selection', 'Selection already sealed.')
        self.frozen_contract = dict(selection=copy.deepcopy(self.selections), hp=copy.deepcopy(self.hp))
        write_json(self.result_dir / 'selection_frozen.json', self.frozen_contract)
        self.phase = 'frozen'
        return pd.DataFrame([dict(experiment=k, prompt=v['prompt_style'],
                                  alpha=v.get('alpha'), preservation=v.get('lambda_preservation'))
                             for k, v in self.selections.items()])

    def _evaluate(self, name, state, style, details):
        require(self.phase == 'frozen', 'Final tests prohibited before validation selection is frozen.')
        strict_load(self.model, state, name + ' evaluation')
        _, _, seen, unseen = self.loaders()
        path = self.result_dir / name
        path.mkdir(exist_ok=False)
        self.model.requires_grad_(False)
        same_text = all(torch.equal(state[k].cpu(), v) for k, v in self.base.items()
                        if not k.startswith('visual.'))
        text = self.text(style) if same_text else encode_groups(
            self.model, self.tokenizer, prompt_groups(self.pairs, self.hp, style),
            self.device, self.hp['text_chunk_size'])
        predict = make_predict(self.model, text, self.pairs)
        with evaluation_directory(path):
            result = evaluate(predict, evaluation_loader(seen), evaluation_loader(unseen), name,
                              dict(details, samples=self.counts, debug=self.debug), robust=self.hp['robust'])
        return dict(status='evaluated', metrics=result, confusion_png=str(path / (name+'_confusion.png')),
                    confusion_csv=str(path / (name+'_confusion.csv')), scope='DEBUG tiny subset' if self.debug else 'official full test')

    def evaluate_all(self):
        require(self.phase == 'frozen', 'Run seal_selection() before evaluation.')
        require(self.hp == self.frozen_contract['hp'] and self.selections == self.frozen_contract['selection'],
                'Settings changed after selection; start a new run.')
        require(not hasattr(self, 'reports'), 'Final tests already ran; refusing overwrite.')
        self.reports = {}
        self.reports['baseline_zero'] = self._evaluate('baseline_zero', self.base, 'single', self.hp)
        self.reports['baseline_partial6'] = self._evaluate('baseline_partial6', self.partial, 'single', self.hp)
        self._full_baseline()
        for name, record in self.selections.items():
            state = self.debug_states[name] if self.debug else extract_state(read_checkpoint(Path(record['checkpoint'])))
            self.reports[name] = self._evaluate(name, state, record['prompt_style'], record)
        require(file_hash(self.original_path) == self.original_hash, 'Original notebook was modified.')
        write_json(self.result_dir / 'comparison.json', dict(metadata=self.metadata(), selections=self.selections, reports=self.reports))
        self.phase = 'complete'
        return self.table()

    def _full_baseline(self):
        path = self.partial_path.with_name('approach3_clip_full.pt')
        try:
            checkpoint = read_checkpoint(path)
            require(torch.equal(checkpoint.get('pairs', torch.empty(0)), self.pairs), 'Full baseline pair order mismatch.')
            self.reports['baseline_full'] = self._evaluate('baseline_full', extract_state(checkpoint), 'single', self.hp)
        except (OSError, ValueError, RuntimeError) as error:
            self.reports['baseline_full'] = dict(status='unavailable', reason=str(error), expected_checkpoint=str(path))
            self._historical_full()

    def _historical_full(self):
        if not self.baseline_results:
            return
        try:
            payload = json.loads(Path(self.baseline_results).read_text(encoding='utf-8'))
            result = next(r for r in payload['results'] if r['model'] == 'approach3_clip_full')
            self.reports['baseline_full']['historical_metrics'] = result
            self.reports['baseline_full']['historical_source'] = str(self.baseline_results)
            self.reports['baseline_full']['note'] = 'Historical record only; checkpoint not reloaded, excluded from DEBUG comparison.'
        except (OSError, ValueError, KeyError, StopIteration) as error:
            self.reports['baseline_full']['historical_error'] = str(error)

    def table(self):
        columns = ['seen_top1', 'seen_top5', 'unseen_top1', 'unseen_top5', 'harmonic_mean', 'crop_acc', 'disease_acc']
        rows = [dict(experiment=name, status=record['status'], **{k: record.get('metrics', {}).get(k) for k in columns})
                for name, record in self.reports.items()]
        return pd.DataFrame(rows)

    def save_table(self):
        require(self.phase == 'complete', 'Finish evaluation before saving the final table.')
        target = self.result_dir / 'comparison.csv'
        with target.open('x', encoding='utf-8', newline='') as stream:
            self.table().to_csv(stream, index=False)
        return target

    def execute(self, method):
        """Record an actionable failure; never shrink a full experiment silently."""
        try:
            return getattr(self, method)()
        except Exception as error:
            path = self.result_dir / (method + '_error.json')
            if not path.exists():
                write_json(path, dict(stage=method, error=type(error).__name__, message=str(error),
                                      corrective_action='Inspect the named contract/setting; start a new run after fixing it.'))
            if isinstance(error, torch.cuda.OutOfMemoryError):
                raise RuntimeError(f'{method}: CUDA OOM with batch={self.hp["batch_size"]}, '
                                   f'precision={self.hp["precision"]}. No protocol was reduced. '
                                   'Use a larger GPU or declare new settings for every comparison.') from error
            raise

    def close(self):
        del self.model, self.teacher, self.base, self.partial
        self.text_bank.clear()
        self.debug_states.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
