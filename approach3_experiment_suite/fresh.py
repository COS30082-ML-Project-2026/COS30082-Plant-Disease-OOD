"""Fresh initialization, training-only development, separately gated final test."""
import copy
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import random
import torch
from common import config as shared
from approach3_ood_backbones.adapters import load_adapter
from approach3_ood_backbones.experiments import collect, new_run, write_json, score_report, save_confusion
from approach3_ood_backbones.factorized import combine, standardize
from approach3_ood_backbones.ood_evaluation import top1_predictions
from approach3_experiment_suite.protocol import METHODS, validate_config, class_space, bias_scores
from approach3_experiment_suite.engine import train, model_scores, inference
from approach3_experiment_suite.models import build_model
from approach3_experiment_suite.fresh_data import development, final_data, sha

TRAINED = {'R3', 'R4', 'F3', 'F4'}


def seed():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)


def release():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def banks_for(adapter, pairs, names, config, symptoms=None):
    crops, diseases, columns = class_space(pairs)
    crop_names, disease_names = names
    clean = lambda text: text.replace('_', ' ')
    prompts = []
    for crop, disease in pairs.tolist():
        text = config['pair_prompt'].format(crop=clean(crop_names[crop]), disease=clean(disease_names[disease]))
        row = (symptoms or {}).get('descriptions', {}).get(str(disease))
        if row:
            if row['label'] != disease_names[disease] or not row['source'].startswith('https://'):
                raise ValueError('Symptom label/source mismatch.')
            text += '; ' + row['text']
        prompts.append(text)
    groups = [prompts, [config['crop_prompt'].format(crop=clean(crop_names[c])) for c in crops],
              [config['disease_prompt'].format(disease=clean(disease_names[d])) for d in diseases]]
    size = config['text_batch_size']
    return [torch.cat([adapter.encode_text(group[i:i+size]) for i in range(0, len(group), size)])
            for group in groups], columns, groups


def choose_weights(components, columns, labels, grid):
    labels = torch.as_tensor(labels).long()
    history = [dict(weights=list(weights), correct=int((top1_predictions(combine(components, columns, weights)) == labels).sum()))
               for weights in grid]
    winner = max(history, key=lambda row: row['correct'])
    return dict(weights=winner['weights'], history=history,
                source='training-only compositional validation; 323 training-derived candidates; topk(1)')


def choose_bias(a, al, b, bl, seen_rows, grid):
    history = []
    for bias in grid:
        x = (top1_predictions(bias_scores(a, seen_rows, bias)) == torch.as_tensor(al)).float().mean().item()
        y = (top1_predictions(bias_scores(b, seen_rows, bias)) == torch.as_tensor(bl)).float().mean().item()
        history.append(dict(bias=bias, harmonic_mean=2*x*y/(x+y) if x+y else 0.))
    winner = max(history, key=lambda row: row['harmonic_mean'])
    return dict(bias=winner['bias'], history=history, source='training-only proxy holdout; config-order tie break')


def identity(metadata):
    return {key: metadata.get(key) for key in ('weight_sha256', 'resolved_revision', 'preprocessing', 'tokenizer', 'scoring')}


def expand_model(model, method, pairs, banks):
    """Only add final candidate queries; never change any learned parameter."""
    before = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}
    if method in ('F3', 'F4'):
        for i, bank in enumerate(banks): setattr(model, 'bank' + str(i), bank.detach())
    else:
        _, _, columns = class_space(pairs)
        if int(columns[:, 0].max()) + 1 != model.nc or int(columns[:, 1].max()) + 1 != model.nd:
            raise ValueError('Final components differ from training vocabulary.')
        model.columns = columns.to(model.device)
    if any(not torch.equal(value, dict(model.named_parameters())[name]) for name, value in before.items()):
        raise RuntimeError('Candidate expansion changed learned weights.')


class FreshSuite:
    def __init__(self, config, mode='smoke', allow_download=False, output=None):
        validate_config(config)
        if config.get('protocol') != 'train_only_v2': raise ValueError('Use fresh_config.json: lecturer train-only protocol.')
        if any(key in config for key in ('e1_path', 'checkpoint_dir', 'score_caches', 'resume')):
            raise ValueError('Old model/checkpoint/cache inputs are forbidden in the fresh suite.')
        if config['backbones'] != {
                'clip_b16': dict(backend='open_clip', architecture='ViT-B-16', pretrained='openai'),
                'siglip2': dict(backend='transformers', repository='google/siglip2-base-patch16-224', revision='main')}:
            raise ValueError('Only the declared public pretrained backbones are allowed.')
        if config['dino_repository'] != 'facebook/dinov2-base' or config['dino_pooling'] != 'last_hidden_state[:, 0, :] (CLS)':
            raise ValueError('Keep the declared public DINO model and CLS pooling.')
        if not isinstance(config['dino_batch_size'], int) or config['dino_batch_size'] < 1:
            raise ValueError('DINO batch size must be positive.')
        if mode not in ('smoke', 'full'): raise ValueError('Choose smoke or full.')
        if mode == 'full' and not torch.cuda.is_available(): raise RuntimeError('Full runs require CUDA.')
        self.config = copy.deepcopy(config); self.mode = mode; self.allow_download = allow_download
        if mode == 'smoke': self.config.update(epochs=1, batch_size=2, accumulation_steps=1, precision='float32')
        self._frozen_config = copy.deepcopy(self.config)
        self.dev = development(config['pair_holdout_fraction'])
        self.path = new_run('fresh_' + mode, output or shared.RESULTS_DIR / 'approach3_suite_fresh')
        self.completed = {}
        write_json(self.path / 'requested_config.json', config)
        write_json(self.path / 'effective_config.json', self.config)
        write_json(self.path / 'development_audit.json', dict(mode=mode, candidates=self.dev['pairs'].tolist(),
            train_csv_sha256=self.dev['train_csv_sha256'], signatures=self.dev['signatures'], split=self.dev['manifest'],
            official_test_csvs_read=False, initialization='public pretrained weights only; no previous user checkpoints',
            final_retraining_on_7223=False, neural_training_partition='reduced partition; see split counts'))

    def catalog(self):
        return pd.DataFrame([dict(id=k, name=('Pristine CLIP + frozen DINO crop fusion' if k == 'D3' else v[0]),
                                  selected=k in self.config['selected']) for k, v in METHODS.items()])

    def _adapter(self, method, spec=None):
        spec = copy.deepcopy(spec or self.config['backbones']['siglip2' if method in ('F2', 'F3', 'F4') else 'clip_b16'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu': torch.set_num_threads(4)
        if device.type == 'cuda' and self.config['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 unsupported; choose float32 before development.')
        return load_adapter(spec, device, self.allow_download)

    def run(self, method):
        if self.config != self._frozen_config: raise RuntimeError('Settings changed after session creation; start a new session.')
        if method not in self.config['selected']: raise ValueError('Method is not selected.')
        if method in self.completed or (self.path / 'development_sealed.json').exists(): raise RuntimeError('Already started or sealed.')
        target = self.path / method; target.mkdir(exist_ok=False)
        self.completed[method] = dict(status='started')
        seed(); adapter = None; model = None
        try:
            adapter = self._adapter(method)
            pairs = self.dev['pairs']; config = self.config
            banks, columns, prompts = banks_for(adapter, pairs, self.dev['names'], config)
            write_json(target / 'development_prompts.json', prompts)
            selection = {}; symptoms = None
            validation = self.dev['holdout'].iloc[:2] if self.mode == 'smoke' else self.dev['holdout']
            reduced = self.dev['reduced'].iloc[:2] if self.mode == 'smoke' else self.dev['reduced']
            if method in TRAINED:
                model, selection = train(method, adapter, banks, pairs, config, reduced, validation, target,
                                         self.mode == 'smoke', name_maps=self.dev['names'])
                if method == 'F4':
                    components = model_scores(model, adapter, validation, config, True)
                    selection['factorized'] = choose_weights(components, columns, validation.pair_row.to_numpy(), config['score_grid'])
            elif method == 'F2':
                components = collect(adapter, banks, validation, config['batch_size'])
                selection['factorized'] = choose_weights(components, columns, validation.pair_row.to_numpy(), config['score_grid'])
            elif method == 'R1':
                seen = self.dev['validation'][self.dev['validation'].pair_row.isin(self.dev['reduced'].pair_row)]
                if self.mode == 'smoke': seen = seen.iloc[:2]
                a = collect(adapter, banks[:1], seen, config['batch_size'])[0]
                b = collect(adapter, banks[:1], validation, config['batch_size'])[0]
                selection = choose_bias(a, seen.pair_row.to_numpy(), b, validation.pair_row.to_numpy(),
                                        sorted(set(self.dev['reduced'].pair_row)), config['bias_grid'])
            elif method == 'R5':
                symptoms = json.loads((Path(__file__).parent / 'symptoms.json').read_text(encoding='utf8'))
                _, _, prompts = banks_for(adapter, pairs, self.dev['names'], config, symptoms)
                write_json(target / 'symptoms_frozen.json', symptoms)
            elif method == 'D3':
                # This is a NEW pristine-CLIP experiment, not the former E1 fusion.
                from approach3_dino_fusion.core import Dino, prototypes
                dino = Dino(config, adapter.device, self.allow_download)
                source = self.dev['reduced']
                if self.mode == 'smoke': source = source.groupby('crop', sort=True).head(1)
                ids = torch.tensor(sorted(set(source.crop)))
                features = dino.embeddings(source, config['dino_batch_size'])
                centers, counts = prototypes(features, source.crop.to_numpy(), ids)
                torch.save(dict(centers=centers, ids=ids, counts=counts, metadata=dino.metadata,
                                prototype_images=source.image.tolist()), target / 'crop_prototypes.pt')
                selection = dict(clip_fraction=.8, initialization='pristine public CLIP, NEVER E1', prototype_images=len(source))
                del dino
            metadata = adapter.metadata
            spec = copy.deepcopy(metadata['spec'])
            if spec['backend'] == 'transformers': spec['revision'] = metadata['resolved_revision']
            write_json(target / 'prepared.json', dict(method=method, selection=selection, spec=spec,
                backbone_identity=identity(metadata), model_metadata=metadata, mode=self.mode,
                candidates=323, official_test_csvs_read=False,
                training_images=len(reduced) if model is not None else 0))
            self.completed[method] = dict(status='prepared')
            return target
        except Exception as error:
            write_json(target / 'error.json', dict(type=type(error).__name__, message=str(error)))
            self.completed[method] = dict(status='failed')
            raise
        finally:
            del model, adapter; release()

    def run_selected(self):
        for method in self.config['selected']: self.run(method)
        return self.save()

    def save(self):
        if self.config != self._frozen_config: raise RuntimeError('Settings changed after session creation; cannot seal.')
        if set(self.completed) != set(self.config['selected']) or any(v['status'] != 'prepared' for v in self.completed.values()):
            raise RuntimeError('Every selected method must finish development before sealing.')
        artifacts = ['effective_config.json', 'development_audit.json']
        for method in self.config['selected']:
            artifacts.extend([method + '/prepared.json', method + '/development_prompts.json'])
            if method in TRAINED: artifacts.append(method + '/best.pt')
            if method == 'R5': artifacts.append(method + '/symptoms_frozen.json')
            if method == 'D3': artifacts.append(method + '/crop_prototypes.pt')
        write_json(self.path / 'development_sealed.json', dict(artifact_sha256={p: sha(self.path / p) for p in artifacts},
            selected=self.config['selected'], mode=self.mode, train_csv_sha256=self.dev['train_csv_sha256'],
            signatures=self.dev['signatures'], official_test_csvs_read=False))
        return self.path


def final_evaluate(development_path, allow_download=False):
    root = Path(development_path)
    seal = json.loads((root / 'development_sealed.json').read_text())
    for relative, expected in seal['artifact_sha256'].items():
        if sha(root / relative) != expected: raise ValueError('Sealed development artifact changed: ' + relative)
    config = json.loads((root / 'effective_config.json').read_text())
    if config['protocol'] != 'train_only_v2' or config['selected'] != seal['selected']:
        raise ValueError('Wrong frozen protocol/methods.')
    dev = development(config['pair_holdout_fraction'])
    if dev['train_csv_sha256'] != seal['train_csv_sha256'] or dev['signatures'] != seal['signatures']:
        raise ValueError('Training data changed after development.')
    path = new_run('final_' + seal['mode'], root / 'evaluations')
    write_json(path / 'evaluation_started.json', dict(seal_sha256=sha(root / 'development_sealed.json'),
        settings_frozen=True, note='No hyperparameter selection or gradient updates in this stage.'))
    # First official test CSV access occurs HERE, after all artifacts are checked.
    if seal['mode'] == 'smoke':
        pairs = dev['pairs']
        frames = dict(train=dev['reduced'], validation=dev['validation'],
                      seen=dev['validation'].iloc[:2], unseen=dev['holdout'].iloc[:2])
    else:
        if not torch.cuda.is_available(): raise RuntimeError('Full inference requires CUDA.')
        pairs, frames = final_data(dev)
    write_json(path / 'candidate_space.json', dict(pairs=pairs.tolist(), count=len(pairs),
        official_test_read=seal['mode'] == 'full', smoke=seal['mode'] == 'smoke',
        frame_images={k: frames[k].image.tolist() for k in ('seen', 'unseen')}))
    table = []
    for method in seal['selected']:
        source = root / method; target = path / method; target.mkdir()
        seed(); adapter = model = dino = None
        try:
            prepared = json.loads((source / 'prepared.json').read_text())
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            adapter = load_adapter(prepared['spec'], device, allow_download)
            if identity(adapter.metadata) != prepared['backbone_identity']:
                raise ValueError('Public pretrained backbone/preprocessing changed after selection.')
            banks, columns, prompts = banks_for(adapter, pairs, dev['names'], config)
            if method in TRAINED:
                old_banks, _, _ = banks_for(adapter, dev['pairs'], dev['names'], config)
                model = build_model(method, adapter, old_banks, dev['pairs'], config, name_maps=dev['names']).to(device)
                saved = torch.load(source / 'best.pt', map_location='cpu', weights_only=True)
                if not torch.equal(saved['pairs'], dev['pairs']): raise ValueError('Checkpoint development candidates changed.')
                model.load_state_dict(saved['state_dict'], strict=True)
                del saved
                expand_model(model, method, pairs, banks)
                model.eval().requires_grad_(False)
            symptoms = None
            if method == 'R5':
                symptoms = json.loads((source / 'symptoms_frozen.json').read_text())
                symptom_banks, _, prompts = banks_for(adapter, pairs, dev['names'], config, symptoms)
            if method == 'D3':
                from approach3_dino_fusion.core import Dino
                proto = torch.load(source / 'crop_prototypes.pt', map_location='cpu', weights_only=True)
                frozen_dino = copy.deepcopy(config); frozen_dino['dino_revision'] = proto['metadata']['resolved_revision']
                dino = Dino(frozen_dino, device, allow_download)
                if identity(dino.metadata) != identity(proto['metadata']): raise ValueError('DINO provenance changed.')
                crop_map = {int(c): i for i, c in enumerate(proto['ids'])}
                crop_columns = torch.tensor([crop_map[int(c)] for c in pairs[:, 0]])
            write_json(target / 'final_prompts.json', prompts)
            predictions = {}; baseline = {}
            with torch.inference_mode():
                for split in ('seen', 'unseen'):
                    frame = frames[split]
                    components = model_scores(model, adapter, frame, config, method in ('F3', 'F4')) if model is not None else collect(adapter, banks, frame, config['batch_size'])
                    baseline[split] = components[0]
                    selection = prepared['selection']
                    if method in TRAINED: scores = inference(method, components, columns, selection.get('factorized', {}).get('weights'))
                    elif method == 'F2': scores = combine(components, columns, selection['factorized']['weights'])
                    elif method == 'R1': scores = bias_scores(components[0], range(323), selection['bias'])
                    elif method == 'R5': scores = collect(adapter, symptom_banks[:1], frame, config['batch_size'])[0]
                    else:
                        features = dino.embeddings(frame, config['dino_batch_size'])
                        cosine = features @ proto['centers'].T
                        scores = .8 * standardize(components[0]) + .2 * standardize(cosine)[:, crop_columns]
                    if scores.shape != (len(frame), len(pairs)) or not torch.isfinite(scores).all():
                        raise ValueError('Invalid final score matrix.')
                    predictions[split] = scores
            for variant, bundle in [('pair_or_name_only', baseline), (method, predictions)]:
                report = score_report(bundle['seen'], bundle['unseen'], frames, pairs)
                report.update(method=method, variant=variant, smoke=seal['mode'] == 'smoke',
                    candidate_space=f'all {len(pairs)} candidates; expanded only after development sealing',
                    protocol='train_only_v2', training_images=prepared['training_images'],
                    limitation='Reduced-training experiment; no retraining on all 7223. Historical test exposure remains; exploratory.',
                    robustness={'status': 'not_run; clean evaluation only'})
                write_json(target / (variant + '_report.json'), report)
                table.append(dict(method=method, variant=variant, **{k: report[k] for k in
                    ('seen_top1', 'seen_top5', 'unseen_top1', 'unseen_top5', 'harmonic_mean', 'crop_acc', 'disease_acc')}))
            torch.save(dict(scores=predictions, baseline=baseline, pairs=pairs,
                labels={k: torch.tensor(frames[k].pair_row.to_numpy()) for k in predictions},
                images={k: frames[k].image.tolist() for k in predictions}, smoke=seal['mode'] == 'smoke'), target / 'scores.pt')
            if seal['mode'] == 'full': save_confusion(target, method, predictions['seen'], frames['seen'], pairs)
        except Exception as error:
            write_json(target / 'error.json', dict(type=type(error).__name__, message=str(error)))
            raise
        finally:
            del model, adapter, dino; release()
    pd.DataFrame(table).to_csv(path / 'all_comparisons.csv', index=False)
    write_json(path / 'completed.json', dict(methods=seal['selected'], smoke=seal['mode'] == 'smoke'))
    return path
