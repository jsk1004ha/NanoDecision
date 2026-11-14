"""Explicit experiments, not hardware auto-tuning or claimed optimal settings."""
import copy


def apply_profile(recipe,name):
    if name not in ('throughput64','batch128','batch256','context4k','context8k'):raise ValueError('Unknown training profile')
    result=copy.deepcopy(recipe)
    if result.get('smoke_only') and name.startswith('context'):
        raise ValueError('Byte fixture position limits cannot certify a 4K/8K profile')
    effective={'throughput64':64,'batch128':128,'batch256':256,'context4k':64,'context8k':64}[name]
    result['training'].update(trainer='scalable',batch_size=4 if not name.startswith('context') else 1,
        accumulation=effective//(4 if not name.startswith('context') else 1),
        workers=2,prefetch=2,pin_memory=True,rdrop=0.,max_tokens=0,ddp_world=1)
    if name=='context4k':result['max_state_tokens']=4096
    if name=='context8k':result['max_state_tokens']=8192
    result['training_profile']=name
    result.setdefault('notes',[]).append('Explicit experimental profile. Probe device memory first; no truncation or measured speed/quality claim.')
    return result
