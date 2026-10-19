#!/usr/bin/env python3
"""Cache a separate corpus, then optionally adapt the encoder with masked LM."""
import argparse
import json
from nanodecision2.adaptation import prepare_corpus,adapt


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    c=sub.add_parser("prepare");c.add_argument("--input",required=True);c.add_argument("--init",required=True);c.add_argument("--out",required=True);c.add_argument("--block-length",type=int,default=256)
    c=sub.add_parser("train")
    for x in ("init","corpus","out"):c.add_argument("--"+x,required=True)
    c.add_argument("--device",choices=("cpu","cuda"),default="cpu");c.add_argument("--epochs",type=int,default=1)
    c.add_argument("--batch-size",type=int,default=2);c.add_argument("--lr",type=float,default=1e-5)
    c.add_argument("--mask-chunk",type=int,default=32);c.add_argument("--mask-token-id",type=int)
    c.add_argument("--max-steps",type=int,default=0);c.add_argument("--seed",type=int,default=42);c.add_argument("--threads",type=int,default=2)
    c.add_argument("--bf16",action="store_true");c.add_argument("--gradient-checkpointing",action="store_true")
    a=p.parse_args()
    result=prepare_corpus(a.input,a.out,a.init,a.block_length) if a.command=="prepare" else adapt(a)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
