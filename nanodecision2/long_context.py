"""Experimental all-block C++ context. Not full-attention 32K or a quality claim.

Every supplied block is encoded. The trained decision head attends to all
concatenated key/value features, rather than averaging independent probabilities.
Only separately fitted all-block calibration is eligible for use.
"""
import argparse
import json
from pathlib import Path
from .runtime import DecisionEngine
from .schema import normalize_question
from .probability import softmax
from .decision_contract import calibration_language,decision_result
from .multilingual import served_temperature


class AllBlocksEngine:
    def __init__(self,bundle,block_tokens=None,max_tokens=32768,allow_smoke=False):
        self.engine=DecisionEngine(bundle,allow_smoke=allow_smoke)
        approved=self.engine.manifest['max_state_tokens']
        self.block_tokens=approved if block_tokens is None else block_tokens
        if type(self.block_tokens)is not int or not 1<=self.block_tokens<=approved:
            raise ValueError('Block size exceeds the existing direct-input contract')
        if type(max_tokens)is not int or not self.block_tokens<=max_tokens<=32768 or (max_tokens+self.block_tokens-1)//self.block_tokens>256:
            raise ValueError('Invalid combined context budget')
        self.max_tokens=max_tokens;self.calibration=None
        path=Path(bundle)/'all_blocks_calibration.json'
        if path.exists():
            c=json.loads(path.read_text())
            expected={'model_sha256':self.engine.manifest['model_sha256'],'context_mode':'all_blocks','block_tokens':self.block_tokens}
            if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('All-block calibration contract mismatch')
            softmax([0.,0.],c['temperature']);self.calibration=c

    def decide(self,state,questions,language=None):
        if not isinstance(questions,dict) or not 1<=len(questions)<=64:raise ValueError('Expected 1..64 questions')
        ids=self.engine.codec.state(state,self.max_tokens)
        blocks=[ids[i:i+self.block_tokens] for i in range(0,len(ids),self.block_tokens)]
        context=self.engine.native.prepare_many(self.engine.handle,blocks)
        answers={}
        for name,q in questions.items():
            spec=normalize_question(q)
            pairs=self.engine.codec.pairs(spec,self.engine.manifest['max_pair_tokens'])
            z=self.engine.native.score(self.engine.handle,context,pairs)
            lang=calibration_language(q,language)
            t=served_temperature(self.calibration,lang,spec['type'],len(z))
            result=decision_result(spec,softmax(z,t))
            result.update(logits=z,temperature_used=t,calibrated=self.calibration is not None,
                context_mode='all_blocks',encoded_tokens=len(ids),block_tokens=self.block_tokens,
                block_offsets=[[i,min(len(ids),i+self.block_tokens)] for i in range(0,len(ids),self.block_tokens)],
                positions_reset_per_block=True,smoke_only=self.engine.smoke_only,
                quality_status='experimental_not_length_quality_certified')
            if self.engine.smoke_only:result['status']='test_only'
            answers[name]=result
        return {'answers':answers,'model_sha256':self.engine.manifest['model_sha256'],'context_mode':'all_blocks'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True);p.add_argument('--input',required=True);p.add_argument('--block-tokens',type=int)
    p.add_argument('--max-tokens',type=int,default=32768);p.add_argument('--allow-smoke',action='store_true')
    a=p.parse_args(argv);r=json.loads(Path(a.input).read_text(encoding='utf-8'))
    engine=AllBlocksEngine(a.bundle,a.block_tokens,a.max_tokens,a.allow_smoke)
    print(json.dumps(engine.decide(r['state'],r['questions'],r.get('language')),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
