import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=="auto":
        from .autopipeline import main as auto_main
        return auto_main(argv[1:])
    p=argparse.ArgumentParser(prog="nd2",description="Native probabilistic decisions; no Jev API is required")
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("init")
    a.add_argument("--out",required=True);a.add_argument("--backbone",default="jhu-clsp/mmBERT-base")
    a.add_argument("--revision",default="main");a.add_argument("--smoke",action="store_true")
    a.add_argument("--skip-upstream-check",action="store_true");a.add_argument("--decision-dim",type=int,default=256)
    a=sub.add_parser("prepare")
    a.add_argument("--input",nargs="+",required=True);a.add_argument("--out",required=True);a.add_argument("--init",required=True)
    a.add_argument("--max-state",type=int,default=2048);a.add_argument("--max-pair",type=int,default=256);a.add_argument("--seed",type=int,default=42)
    a=sub.add_parser("train")
    for name in ("init","cache","out"):a.add_argument("--"+name,required=True)
    a.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    a.add_argument("--epochs",type=int,default=3);a.add_argument("--batch-size",type=int,default=4)
    a.add_argument("--accumulation",type=int,default=16);a.add_argument("--encoder-lr",type=float,default=2e-5)
    a.add_argument("--head-lr",type=float,default=1e-4);a.add_argument("--layer-decay",type=float,default=0.9)
    a.add_argument("--weight-decay",type=float,default=0.01);a.add_argument("--warmup-ratio",type=float,default=0.05)
    a.add_argument("--brier-weight",type=float,default=0.2);a.add_argument("--rps-weight",type=float,default=0.2)
    a.add_argument("--language-weights",type=json.loads,default={},help='JSON, e.g. {"ko":2,"en":1}; changes the training mixture, not the truth labels')
    a.add_argument("--pair-consistency",type=float,default=0.0,help="JS loss for explicitly reviewed bilingual pairs only")
    a.add_argument("--selection-metric",choices=("nll","macro_language_nll","english_nll"),default="nll")
    a.add_argument("--train-language",choices=("en","ko"))
    a.add_argument("--validation-limit",type=int,default=0)
    a.add_argument("--guard-english",action="store_true")
    a.add_argument("--guard-korean",action="store_true")
    a.add_argument("--english-accuracy-drop",type=float,default=0.0)
    a.add_argument("--english-nll-increase",type=float,default=0.0)
    a.add_argument("--eval-every",type=int,default=500);a.add_argument("--save-every",type=int,default=500)
    a.add_argument("--log-every",type=int,default=20);a.add_argument("--patience",type=int,default=6)
    a.add_argument("--seed",type=int,default=42);a.add_argument("--threads",type=int,default=2)
    a.add_argument("--bf16",action="store_true");a.add_argument("--gradient-checkpointing",action="store_true")
    a.add_argument("--resume");a.add_argument("--stop-after",type=int,default=0,help="Stop after a step without altering the planned learning-rate schedule")
    a=sub.add_parser("export")
    a.add_argument("--checkpoint",required=True);a.add_argument("--out",required=True);a.add_argument("--fp32",action="store_true")
    a=sub.add_parser("evaluate")
    for name in ("bundle","cache","out"):a.add_argument("--"+name,required=True)
    a.add_argument("--split",choices=("train","validation","temperature","conformal","test"),default="test")
    a=sub.add_parser("calibrate")
    for name in ("bundle","temperature","conformal"):a.add_argument("--"+name,required=True)
    a.add_argument("--alpha",type=float,default=0.1)
    a.add_argument("--language-calibration",action="store_true")
    a.add_argument("--min-bucket-groups",type=int,default=200)
    a=sub.add_parser("predict")
    a.add_argument("--bundle",required=True);a.add_argument("--input",required=True)
    a.add_argument("--min-confidence",type=float,default=0);a.add_argument("--allow-uncalibrated",action="store_true")
    a=sub.add_parser("stdin")
    a.add_argument("--bundle",required=True);a.add_argument("--allow-uncalibrated",action="store_true")
    args=p.parse_args(argv)
    try:
        if args.command=="init":
            from .initialization import initialize
            result=initialize(args.out,args.backbone,args.revision,args.smoke,args.skip_upstream_check,args.decision_dim)
        elif args.command=="prepare":
            from .data import prepare_cache
            result=prepare_cache(args.input,args.out,args.init,args.max_state,args.max_pair,args.seed)
        elif args.command=="train":
            from .training import train
            result=train(args)
        elif args.command=="export":
            from .training import read_model
            from .export import export_bundle
            info=json.loads((Path(args.checkpoint)/"init.json").read_text())
            result=export_bundle(read_model(args.checkpoint),args.out,args.checkpoint,not args.fp32,info.get("smoke_only",False),
                info.get("max_state_tokens"),info.get("max_pair_tokens",256),info)
        elif args.command=="evaluate":
            from .evaluation import evaluate_cache
            result=evaluate_cache(args.bundle,args.cache,args.split,args.out)
        elif args.command=="calibrate":
            from .evaluation import calibrate_bundle
            result=calibrate_bundle(args.bundle,args.temperature,args.conformal,args.alpha,args.language_calibration,args.min_bucket_groups)
        else:
            from .runtime import DecisionEngine
            engine=DecisionEngine(args.bundle,require_calibration=not args.allow_uncalibrated)
            if args.command=="stdin":
                for line in sys.stdin:
                    try:
                        request=json.loads(line);result=engine.decide(request["state"],request["questions"])
                        print(json.dumps(result,ensure_ascii=False),flush=True)
                    except Exception as e:print(json.dumps({"error":str(e)},ensure_ascii=False),flush=True)
                return
            request=json.loads(Path(args.input).read_text(encoding="utf-8"))
            result=engine.decide(request["state"],request["questions"],args.min_confidence)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    except (ValueError,RuntimeError,FileExistsError,FileNotFoundError) as e:p.exit(2,f"ERROR: {e}\n")

if __name__=="__main__":main()
