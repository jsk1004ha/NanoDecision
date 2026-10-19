"""Stable small validation views and language-filtered training indices."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import numpy as np
from .multilingual import language_key

class DatasetView:
    def __init__(self, dataset, indices):
        self.dataset=dataset;self.indices=np.asarray(indices,dtype=np.int64)
        self.manifest=dataset.manifest;self.root=dataset.root;self.split=dataset.split
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):return self.dataset[int(self.indices[i])]
    def verify(self):self.dataset.verify()
    def close(self):self.dataset.close()


def training_view(dataset,language):
    if not language:return dataset
    path=dataset.root/f'{dataset.split}.{language}.viewidx'
    info_path=path.with_suffix(path.suffix+'.json')
    fp=dataset.manifest['fingerprint']
    if path.exists() and info_path.exists() and json.loads(info_path.read_text()).get('fingerprint')==fp:
        indices=np.memmap(path,dtype='<i8',mode='r')
    else:
        temp=path.with_name(path.name+'.tmp');n=0
        with temp.open('wb') as f:
            for i in range(len(dataset)):
                if language_key(dataset[i].get('language'))==language:
                    f.write(int(i).to_bytes(8,'little',signed=True));n+=1
                if i and i%100000==0:print(json.dumps({'stage':'index_language','processed':i,'selected':n}),flush=True)
        if not n:raise ValueError('No training examples for '+language)
        temp.replace(path);info_path.write_text(json.dumps({'fingerprint':fp,'count':n}))
        indices=np.memmap(path,dtype='<i8',mode='r')
    return DatasetView(dataset,indices)


def validation_view(dataset,limit,seed):
    if not limit or len(dataset)<=limit:return dataset
    groups=defaultdict(list)
    for i in range(len(dataset)):
        r=dataset[i];key=(language_key(r.get('language')),r.get('task',r['type']))
        score=hashlib.sha256((str(seed)+r['id']).encode()).digest()
        groups[key].append((score,i))
    # Round-robin language/task cells, not the first N contiguous examples.
    groups={k:iter(i for _,i in sorted(v)) for k,v in sorted(groups.items())}
    selected=[]
    while groups and len(selected)<limit:
        for key in list(groups):
            try:selected.append(next(groups[key]))
            except StopIteration:del groups[key]
            if len(selected)>=limit:break
    return DatasetView(dataset,sorted(selected))


def english_gate(candidate,reference,accuracy_drop=0.0,nll_increase=0.0):
    c=candidate.get('by_language',{}).get('en');r=reference.get('by_language',{}).get('en')
    if not c or not r or c['accuracy'] is None or r['accuracy'] is None:
        raise ValueError('English guard needs English hard-label validation examples')
    return c['accuracy']>=r['accuracy']-accuracy_drop-1e-12 and c['nll']<=r['nll']+nll_increase+1e-12


def korean_gate(candidate, reference):
    c=candidate.get('by_language',{}).get('ko');r=reference.get('by_language',{}).get('ko')
    if not c or not r or c['accuracy'] is None or r['accuracy'] is None:
        raise ValueError('Korean guard needs Korean hard-label validation examples')
    return c['accuracy']>=r['accuracy']-1e-12 and c['nll']<=r['nll']+1e-12


def bilingual_gate(candidate, reference):
    return english_gate(candidate,reference) and korean_gate(candidate,reference)
