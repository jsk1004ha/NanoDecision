"""Opt-in PyTorch research graph: typed numeric inputs, all-block text, token fusion.

Legacy text bundles and native graphs are untouched. This graph has its own
architecture_id and must never be exported as an existing C++ model.nd2 file.
"""
from dataclasses import asdict,dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model import DecisionModel
from .modalities import structured_tokens,sensor_tokens
from .schema import normalize_question,state_text

ARCHITECTURE_ID='nd2-extended-research-v1'

@dataclass
class ExtensionConfig:
    structured: bool=False
    sensors: bool=False
    token_interaction: bool=False
    evidence: bool=False
    relation_classes: int=0
    patch_size: int=16
    max_cells: int=512
    max_frames: int=8192
    max_channels: int=32
    max_fused_tokens: int=32768
    block_tokens: int=2048
    max_pair_tokens: int=512
    metadata_tokens: int=256
    def __post_init__(self):
        for name in ('patch_size','max_cells','max_frames','max_channels','max_fused_tokens','block_tokens','max_pair_tokens','metadata_tokens'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:raise ValueError('Invalid extension limit '+name)
        if type(self.relation_classes)is not int or not 0<=self.relation_classes<=256:raise ValueError('Invalid relation_classes')
        if self.max_fused_tokens>32768:raise ValueError('Combined research context is capped at 32768 tokens')


class NumericEncoder(nn.Module):
    def __init__(self,features,hidden):
        super().__init__();self.projection=nn.Linear(features,hidden);self.position=nn.Linear(1,hidden,bias=False)
        self.row_mix=nn.Linear(hidden,hidden,bias=False)
        self.mix=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.LayerNorm(hidden))
    def forward(self,text,features,positions,groups=None):
        tokens=text+self.projection(features)+self.position(positions[:,None])
        if groups is not None:
            # Preserve which fields belong to a row without making unordered
            # row position a learned category or losing cross-column relations.
            means={g:tokens[[i for i,x in enumerate(groups) if x==g]].mean(0) for g in set(groups)}
            tokens=tokens+self.row_mix(torch.stack([means[g] for g in groups]))
        return self.mix(tokens)


class ExtendedDecisionModel(nn.Module):
    architecture_id=ARCHITECTURE_ID
    def __init__(self,cfg,extension):
        super().__init__();self.cfg=cfg;self.extension=extension;self.core=DecisionModel(cfg)
        d,h=cfg.hidden_size,cfg.decision_dim
        if max(extension.block_tokens,extension.max_pair_tokens,extension.metadata_tokens)>cfg.max_position_embeddings:
            raise ValueError('Direct block/pair/metadata limit exceeds backbone positions')
        if extension.structured:self.structured_encoder=NumericEncoder(8,d)
        if extension.sensors:self.sensor_encoder=NumericEncoder(3*extension.patch_size,d)
        if extension.token_interaction:
            self.interaction=nn.Linear(2*h,h);self.interaction_norm=nn.LayerNorm(h)
        if extension.evidence:
            self.evidence_query=nn.Linear(d,d,bias=False);self.evidence_start=nn.Linear(d,1);self.evidence_end=nn.Linear(d,1)
        if extension.relation_classes:self.relation_head=nn.Linear(2*d,extension.relation_classes)

    def _encode(self,ids):
        device=next(self.parameters()).device
        ids=torch.tensor([ids],dtype=torch.long,device=device)
        return self.core.encoder(ids,torch.ones_like(ids,dtype=torch.bool))[0]

    def _text(self,text,codec):
        ids=codec.encode('STATE\n'+text,self.extension.max_fused_tokens)
        blocks=[ids[i:i+self.extension.block_tokens] for i in range(0,len(ids),self.extension.block_tokens)]
        hidden=torch.cat([self._encode(block) for block in blocks])
        # Offsets describe token positions, not independently predicted blocks.
        return hidden,{'context_mode':'direct' if len(blocks)==1 else 'all_blocks',
                       'encoded_tokens':len(ids),'blocks':len(blocks),'positions_reset_per_block':len(blocks)>1}

    def context(self,state,codec):
        kind=state.get('modality') if isinstance(state,dict) else 'text'
        if kind not in ('records','table','timeseries'):
            if kind not in (None,'text'):raise ValueError('Unsupported modality')
            text=state.get('text') if isinstance(state,dict) and kind=='text' else state_text(state)
            if not isinstance(text,str):raise ValueError('Text payload must be a string')
            return self._text(text,codec)
        if kind=='timeseries':
            if not self.extension.sensors:raise ValueError('Sensor module is not in this checkpoint')
            encoded=sensor_tokens(state,self.extension.patch_size,self.extension.max_frames,self.extension.max_channels,
                                  self.extension.max_fused_tokens)
            module=self.sensor_encoder
        else:
            if not self.extension.structured:raise ValueError('Structured module is not in this checkpoint')
            encoded=structured_tokens(state,self.extension.max_cells);module=self.structured_encoder
        device=next(self.parameters()).device
        # Deduplicate names within this forward only. No stale training cache.
        lookup={};texts=[]
        for text in encoded['texts']:
            if text not in lookup:lookup[text]=self._encode(codec.encode(text,self.extension.metadata_tokens)).mean(0)
            texts.append(lookup[text])
        context=module(torch.stack(texts),torch.tensor(encoded['features'],device=device,dtype=torch.float32),
                       torch.tensor(encoded['positions'],device=device,dtype=torch.float32),encoded.get('groups'))
        return context,{'context_mode':kind,'encoded_tokens':len(context),'blocks':1,
                        'observed_count':encoded.get('observed_count')}

    def forward(self,state,question,codec,return_aux=False):
        spec=normalize_question(question);context,details=self.context(state,codec)
        if len(context)>self.extension.max_fused_tokens:raise ValueError('Fused context budget exceeded')
        keys=self.core.head.key(context);values=self.core.head.value(context)
        candidate_hidden=[self._encode(ids) for ids in codec.pairs(spec,self.extension.max_pair_tokens)]
        vectors=[]
        for hidden in candidate_hidden:
            projected=self.core.head.candidate(hidden)
            if self.extension.token_interaction:
                attention=(projected@keys.T/math.sqrt(keys.shape[-1])).softmax(-1)@values
                projected=self.interaction_norm(projected+F.gelu(self.interaction(torch.cat([projected,attention],-1))))
            vectors.append(projected.mean(0))
        mask=torch.ones(1,len(context),dtype=torch.bool,device=context.device)
        options=torch.ones(1,len(vectors),dtype=torch.bool,device=context.device)
        logits=self.core.head(keys[None],values[None],mask,torch.stack(vectors)[None],options)[0]
        if not return_aux:return logits
        aux={}
        question_hidden=torch.cat(candidate_hidden).mean(0)
        if self.extension.evidence:
            conditional=torch.tanh(context+self.evidence_query(question_hidden))
            aux['start']=self.evidence_start(conditional).squeeze(-1);aux['end']=self.evidence_end(conditional).squeeze(-1)
        if self.extension.relation_classes:
            aux['relation']=self.relation_head(torch.cat([context.mean(0),question_hidden]))
        return logits,aux,details


def auxiliary_terms(aux,supervision):
    """Missing supervision contributes no loss, never a negative evidence label.

    evidence_token_span indexes the *encoded* STATE-prefixed token sequence;
    only text contexts are accepted by the training workflow for this target.
    """
    losses={}
    span=supervision.get('evidence_token_span')
    if span is not None:
        if 'start' not in aux:raise ValueError('Evidence labels require the configured training head')
        if not isinstance(span,list) or len(span)!=2 or any(type(x)is not int for x in span) or not 0<=span[0]<=span[1]<len(aux['start']):raise ValueError('Invalid evidence token span')
        for key,index in zip(('start','end'),span):
            losses[key]=F.cross_entropy(aux[key][None].float(),torch.tensor([index],device=aux[key].device))
    if 'relation_id' in supervision:
        if 'relation' not in aux:raise ValueError('Relation labels require configured relation classes')
        index=supervision['relation_id']
        if type(index)is not int or not 0<=index<len(aux['relation']):raise ValueError('Invalid relation target')
        losses['relation']=F.cross_entropy(aux['relation'][None].float(),torch.tensor([index],device=aux['relation'].device))
    return losses


def auxiliary_loss(aux,supervision):
    terms=auxiliary_terms(aux,supervision)
    return sum(terms.values()) if terms else None
