"""Single notebook/CLI controller. Old packages remain unchanged."""
import copy
import json
from pathlib import Path

import pandas as pd
import torch

from common import config as shared
from approach3_ood_backbones.experiments import new_run,write_json,audit,run as run_f2
from approach3_dino_fusion.run import execute as run_d3
from approach3_experiment_suite.protocol import METHODS,validate_config
from approach3_experiment_suite.engine import execute_native


class Suite:
    def __init__(self,config,mode='smoke',allow_download=False,output=None):
        validate_config(config)
        if mode not in ('smoke','full'):raise ValueError('Choose smoke or full.')
        if mode=='full' and not torch.cuda.is_available():raise RuntimeError('Full runs require school CUDA GPU.')
        self.config=copy.deepcopy(config);self.mode=mode;self.allow_download=allow_download
        if mode=='smoke':
            self.config.update(epochs=1,batch_size=2,accumulation_steps=1,precision='float32')
        self.path=new_run('suite_'+mode,output or shared.RESULTS_DIR/'approach3_suite')
        self.completed={}
        write_json(self.path/'requested_config.json',config)
        write_json(self.path/'effective_config.json',dict(config=self.config,mode=mode,
            smoke_overrides='epochs1, batch2, accumulation1, float32; formal settings preserved in requested_config.json' if mode=='smoke' else None))
        _,self.audit=audit(dict(pair_holdout_fraction=config['pair_holdout_fraction'],seed=42),self.path/'audit')

    def catalog(self):
        return pd.DataFrame([dict(id=k,name=v[0],type=v[1],selected=k in self.config['selected']) for k,v in METHODS.items()])

    def run(self,method):
        if method not in METHODS:raise ValueError('Unselected/nonrecommended method ID.')
        if method in self.completed:raise RuntimeError('Already ran this method; create a new Suite to repeat.')
        target=self.path/method;target.mkdir(exist_ok=False)
        self.completed[method]=dict(status='started',path=str(target))
        write_json(target/'protocol_frozen.json',dict(method=method,config=self.config,mode=self.mode,
            no_test_hyperparameter_selection=True,original_split_files_unchanged=True))
        try:
            if method=='D3':
                config=json.loads((shared.ROOT_DIR/'approach3_dino_fusion/config.json').read_text())
                config['dino_batch_size']=self.config['dino_batch_size']
                e1=Path(self.config['e1_path'])
                if not e1.is_absolute():e1=shared.ROOT_DIR/e1
                actual=run_d3(config,e1,'full' if self.mode=='full' else 'smoke',self.allow_download,target)
            elif method=='F2':
                config=json.loads((shared.ROOT_DIR/'approach3_ood_backbones/configs/default.json').read_text())
                config.update(batch_size=self.config['batch_size'],text_batch_size=self.config['text_batch_size'])
                actual,_=run_f2(config,'F2',['siglip2'],self.mode=='smoke',self.allow_download,False,target)
            else:
                execute_native(method,self.config,target,self.mode=='smoke',self.allow_download)
                actual=target
            self.completed[method]=dict(status='complete',path=str(actual))
            write_json(target/'status.json',self.completed[method])
            return actual
        except Exception as error:
            self.completed[method]=dict(status='failed',path=str(target),error=type(error).__name__,message=str(error))
            if not (target/'error.json').exists():write_json(target/'error.json',self.completed[method])
            raise

    def run_selected(self):
        for method in self.config['selected']:self.run(method)
        return self.save()

    def save(self):
        if (self.path/'session_index.json').exists():raise RuntimeError('Session already saved; refusing overwrite.')
        write_json(self.path/'session_index.json',self.completed)
        tables=[]
        for method,record in self.completed.items():
            if record['status']!='complete':continue
            path=Path(record['path'])/'comparison.csv'
            if path.is_file():
                table=pd.read_csv(path);table.insert(0,'method_id',method)
                table['source_file']=str(path);table['suite_mode']=self.mode;tables.append(table)
        if tables:pd.concat(tables,ignore_index=True).to_csv(self.path/'all_comparisons.csv',index=False,na_rep='N/A')
        return self.path
