#!/usr/bin/env python3
"""Report Korean/other-language token lengths and exact overflow, no truncation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from nanodecision2.schema import TextCodec,normalize_question,state_text
from nanodecision2.multilingual import language_key
from nanodecision2.runtime import sha256


def audit(init,path,max_state=2048,max_pair=256):
    info=json.loads((Path(init)/'init.json').read_text());codec=TextCodec(Path(init)/'tokenizer.json',smoke=info.get('smoke_only',False))
    buckets=defaultdict(lambda:{'rows':0,'characters':0,'state_tokens':0,'max_state_tokens':0,'state_overflows':0,'candidate_pair_overflows':0})
    def tokens(text):
        return len(text.encode())+2 if codec.smoke else len(codec.tokenizer.encode(text,add_special_tokens=True).ids)
    with open(path,encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line);spec=normalize_question(row);b=buckets[language_key(row.get('language'))];text=state_text(row['state']);n=tokens("STATE\n"+text)
            b['rows']+=1;b['characters']+=len(text);b['state_tokens']+=n;b['max_state_tokens']=max(b['max_state_tokens'],n);b['state_overflows']+=n>max_state
            # Use the actual decision codec so format and two-field overhead match.
            for opt in spec['options']:
                try:codec.pairs({**spec,'options':[opt]},max_pair)
                except ValueError:b['candidate_pair_overflows']+=1
    return {'by_language':dict(buckets),'tokenizer_sha256':info['tokenizer_sha256'],'input_sha256':sha256(path),
            'smoke_only':bool(info.get('smoke_only')),'warning':'Token count is an efficiency diagnostic, not language accuracy. Byte fixtures do not measure the actual mmBERT tokenizer.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('init','input','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--max-state',type=int,default=2048);p.add_argument('--max-pair',type=int,default=256)
    a=p.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError('Output exists')
    out.parent.mkdir(parents=True,exist_ok=True);result=audit(a.init,a.input,a.max_state,a.max_pair)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
