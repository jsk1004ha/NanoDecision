#!/usr/bin/env python3
"""Reconcile exact duplicate states with source groups using disk-backed union/find.

Never merges train and official test/development by silently moving examples.
Cross-language equivalents still require supplied origin group IDs; text hashing
cannot discover translations. No similarity threshold is presented as certainty.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import unicodedata
from nanodecision2.adapters import partition_for
from nanodecision2.schema import state_text
from nanodecision2.runtime import sha256


def reconcile(paths,out,repartition_training=False):
    out=Path(out)
    if out.exists():raise FileExistsError('Output exists')
    out.parent.mkdir(parents=True,exist_ok=True)
    dbpath=Path(str(out)+'.audit.sqlite')
    if dbpath.exists():raise FileExistsError('Audit database already exists')
    db=sqlite3.connect(dbpath)
    db.executescript('CREATE TABLE parent(g TEXT PRIMARY KEY,p TEXT); CREATE TABLE state(h TEXT PRIMARY KEY,g TEXT); CREATE TABLE policy(g TEXT PRIMARY KEY,part TEXT);')
    def root(g):
        trail=[]
        while True:
            p=db.execute('SELECT p FROM parent WHERE g=?',(g,)).fetchone()[0]
            if p==g:break
            trail.append(g);g=p
        for t in trail:db.execute('UPDATE parent SET p=? WHERE g=?',(g,t))
        return g
    def rows():
        for p in paths:
            with open(p,encoding='utf-8') as f:
                for line in f:
                    if line.strip():yield json.loads(line)
    joins=0;count=0
    try:
        for r in rows():
            g=str(r['group_id']);db.execute('INSERT OR IGNORE INTO parent VALUES (?,?)',(g,g))
            h=hashlib.sha256(unicodedata.normalize('NFC',state_text(r['state'])).strip().encode()).hexdigest()
            existing=db.execute('SELECT g FROM state WHERE h=?',(h,)).fetchone()
            if existing:
                a,b=root(g),root(existing[0])
                if a!=b:
                    a,b=sorted((a,b));db.execute('UPDATE parent SET p=? WHERE g=?',(a,b));joins+=1
            else:db.execute('INSERT INTO state VALUES (?,?)',(h,g))
            count+=1
            if count%10000==0:db.commit()
        for r in rows():
            g=root(str(r['group_id']))
            # Only converter-created partitions inside an original training
            # release may be regrouped by a new deterministic hash.
            part='reassign_train' if repartition_training and r.get('source_split')=='train' else r.get('partition')
            if part is None:raise ValueError('Specify explicit partitions before reconciliation')
            old=db.execute('SELECT part FROM policy WHERE g=?',(g,)).fetchone()
            if old and old[0]!=part:raise ValueError('Duplicate/origin group crosses fixed partitions; audit/remove conflicting training rows without changing the held-out source')
            db.execute('INSERT OR IGNORE INTO policy VALUES (?,?)',(g,part))
        tmp=Path(str(out)+'.partial')
        with tmp.open('w',encoding='utf-8') as f:
            for r in rows():
                g=root(str(r['group_id']));part=db.execute('SELECT part FROM policy WHERE g=?',(g,)).fetchone()[0]
                r['original_group_id']=r['group_id'];r['group_id']=g
                r['partition']=partition_for(g,'train') if part=='reassign_train' else part
                # Synthetic probability labels must not certify human-label coverage.
                if r.get('label_origin')=='teacher_distribution_synthetic' and r['partition']=='conformal':
                    r['partition']='validation'
                f.write(json.dumps(r,ensure_ascii=False)+'\n')
        tmp.replace(out);db.commit()
        groups=db.execute('SELECT count(*) FROM policy').fetchone()[0]
        result={'rows':count,'merged_group_links':joins,'groups':groups,'inputs':{str(p):sha256(p) for p in paths},'output_sha256':sha256(out),
                'scope':'Exact NFC-state duplicates and declared origin groups only. Translation and near-duplicate leakage require additional origin metadata/audit.'}
        out.with_suffix('.manifest.json').write_text(json.dumps(result,indent=2));return result
    finally:db.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',nargs='+',required=True);p.add_argument('--out',required=True)
    p.add_argument('--repartition-training',action='store_true')
    a=p.parse_args();print(json.dumps(reconcile(a.input,a.out,a.repartition_training),indent=2))
if __name__=='__main__':main()
