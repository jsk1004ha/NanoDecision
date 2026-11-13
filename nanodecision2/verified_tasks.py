"""Stream deterministic oracle-labelled KO/EN/mixed decision examples.

Synthetic tasks are explicitly labelled as such. More renderings do not create
more independent worlds. They complement, not replace, audited real data.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
from .auto_sources import atomic_json
from .runtime import sha256
from .schema import normalize_question,gold_target

COMBINATIONS=list(itertools.product(('en','ko'),repeat=3))


def partition(group):
    n=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%100
    return 'train' if n<80 else 'validation' if n<90 else 'temperature' if n<95 else 'test'


def oracle(facts, task):
    if task=='recipient':return facts['recipient']
    if task=='actor':return facts['actor']
    if task=='threshold':return facts['reading']>=facts['threshold']
    if task=='verification':return facts['changed'] and facts['test_after_change'] and facts['passed']
    if task=='implementation':return facts['changed']
    raise ValueError('Unknown verified task')


def world(index,seed):
    rng=random.Random(f'{seed}:world:{index}')
    def name():return ''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(12))
    names=[]
    while len(names)<3:
        x=name()
        if x not in names:names.append(x)
    return {'actor':names[0],'recipient':names[1],'other':names[2],
            'reading':rng.randint(-10000,10000)/10,'threshold':rng.randint(-10000,10000)/10,
            'changed':bool(rng.getrandbits(1)),'test_after_change':bool(rng.getrandbits(1)),
            'passed':bool(rng.getrandbits(1))}


def render(facts,task,sl,ql,cl,variant=0,code_switch=False):
    yesno={'false':'거짓','true':'참'} if cl=='ko' else {'false':'False','true':'True'}
    if task in ('recipient','actor'):
        if sl=='ko':text=(f"{facts['actor']}이 {facts['recipient']}에게 도움을 주었다. {facts['other']}은 참여하지 않았다." if not variant else f"도움을 받은 사람은 {facts['recipient']}이고, 도운 사람은 {facts['actor']}이다. {facts['other']}은 별개의 인물이다.")
        else:text=(f"{facts['actor']} helped {facts['recipient']}. {facts['other']} did not participate." if not variant else f"Help was received by {facts['recipient']} from {facts['actor']}; {facts['other']} was not involved.")
        if code_switch:text=f"{facts['actor']}이 {facts['recipient']}를 helped. {facts['other']}은 not involved."
        queries={'recipient':('Who received help?','도움을 받은 사람은 누구인가?'),'actor':('Who provided help?','도움을 준 사람은 누구인가?')}
        q=queries[task][ql=='ko']
        if variant:q=({'recipient':('Select the recipient of the assistance.','지원받은 쪽에 해당하는 인물을 고르시오.'),'actor':('Identify the person providing assistance.','지원을 제공한 쪽의 인물을 고르시오.')}[task][ql=='ko'])
        if code_switch:q=({'recipient':('help를 받은 person은 누구인가?','어느 person이 도움을 received 했는가?'),'actor':('help를 준 person은 누구인가?','어느 person이 도움을 provided 했는가?')}[task][ql=='ko'])
        criteria={v:(f'인물 {v}' if cl=='ko' else f'Person {v}') for v in (facts['other'],facts['recipient'],facts['actor'])}
        return text,{'type':'choice','question':q,'criteria':criteria,'target':oracle(facts,task)}
    if task=='threshold':
        text=(f"센서 {facts['actor']}의 측정값은 {facts['reading']}°C, 기준값은 {facts['threshold']}°C이다." if sl=='ko' else f"Sensor {facts['actor']}: measured temperature {facts['reading']}°C. Threshold: {facts['threshold']}°C.")
        if code_switch:text=f"sensor {facts['actor']}의 temperature는 {facts['reading']}°C, threshold는 {facts['threshold']}°C."
        q=('측정값이 기준값 이상인가?' if ql=='ko' else 'Is the reading greater than or equal to the threshold?')
        if variant:q='기준 온도에 도달했거나 초과했는가?' if ql=='ko' else 'Has the temperature reached or exceeded the threshold?'
    else:
        # These are observed states, not an injected label for the current task.
        text=json.dumps({'component':facts['actor'],'code_changed':facts['changed'],'tests_run_after_change':facts['test_after_change'],
                         'latest_test_passed':facts['passed']},sort_keys=True)
        if sl=='ko':text=f"컴포넌트 {facts['actor']}: 코드 수정: {facts['changed']}; 수정 후 테스트 실행: {facts['test_after_change']}; 마지막 테스트 통과: {facts['passed']}."
        if code_switch:text=f"component {facts['actor']}: code 수정={facts['changed']}, 변경 후 tests 실행={facts['test_after_change']}, tests passed={facts['passed']}"
        if task=='verification':
            q=('완료 기준은 코드 수정, 수정 후 테스트 실행, 테스트 통과를 모두 충족하는 것이다. 완료인가?' if ql=='ko' else 'Completion requires a code change, tests after that change, and passing tests. Is it complete?')
        else:q='코드만 수정하면 완료다. 완료인가?' if ql=='ko' else 'A code change alone defines completion. Is it complete?'
        if variant:
            if task=='verification':q='수정 작업과 수정 이후의 테스트 실행, 해당 테스트 통과가 전부 필요하다. 이 조건들을 충족했는가?' if ql=='ko' else 'Have all three requirements been met: changing code, running tests after the change, and passing them?'
            else:q='테스트 여부와 관계없이 코드가 수정되면 요구사항을 충족한다. 충족했는가?' if ql=='ko' else 'The requirement is satisfied by changing code, regardless of tests. Is it satisfied?'
    if code_switch:
        q={'threshold':('reading이 threshold 이상인가?','측정 temperature가 기준 이상에 도달했나?'),
           'verification':('code 수정, 변경 후 tests 실행, pass를 모두 충족해야 완료다. 완료인가?','완료 requires code change, post-change tests, passing tests. 전부 충족했나?'),
           'implementation':('tests와 관계없이 code 수정만 하면 완료다. 완료인가?','code change alone이면 done이다. 완료했나?')}[task][ql=='ko']
    return text,{'type':'noul','question':q,'criteria':yesno,'target':oracle(facts,task)}


def examples(index,seed=42,fewshot=0):
    facts=world(index,seed);group=f'verified-world:{seed}:{index}';part=partition(group)
    # Family 1 is held out from train; the generator does not treat views as
    # independent random train/test rows. Both opposite-role tasks share world.
    family=0 if part=='train' else 1
    for task in ('recipient','actor','threshold','verification','implementation'):
        rendered=0
        for combo,(sl,ql,cl) in enumerate(COMBINATIONS):
            for mixed in (False,True):
                if mixed and sl=='en':continue  # code-switch text is shared; emit it only once
                state,q=render(facts,task,sl,ql,cl,family,mixed)
                if fewshot:
                    support=[]
                    for k in range(fewshot):
                        sf=world(index*7+k+1,seed+100000)
                        ss,sq=render(sf,task,sl,ql,cl,family,mixed)
                        support.append({'state':ss,'question':sq['question'],'criteria':sq['criteria'],'demonstrated_answer':sq['target']})
                    state={'support_examples':support,'query':state}
                row={'id':f'{group}:{task}:{combo}:{int(mixed)}','group_id':group,'partition':part,
                     'state':state,**q,'task':task,'language':'mixed' if mixed or len({sl,ql,cl})>1 else sl,
                     'state_language':'mixed' if mixed else sl,'question_language':'mixed' if mixed else ql,
                     'candidate_languages':cl,'code_switch':mixed,'template_family_id':f'family-{family}',
                     'label_origin':'program_verified_synthetic','source':'nd2-verified-tasks-v1',
                     'fewshot_examples':fewshot,'synthetic':True}
                row.update(semantic_group_id=f'{group}:{task}',equivalence_id=f'{group}:{task}:pair:{rendered//2}',
                           equivalence_verified=True,equivalence_review='deterministic_oracle_templates_v1')
                rendered+=1
                gold_target(row,normalize_question(row))
                yield row


def generate(out,worlds,seed=42,fewshot=0):
    if type(worlds)is not int or worlds<1 or fewshot not in (0,1,4):raise ValueError('Positive worlds and 0/1/4-shot required')
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_name(out.name+'.partial')
    counts={};languages={};n=0
    with tmp.open('x',encoding='utf-8') as handle:
        for i in range(worlds):
            for row in examples(i,seed,fewshot):
                handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');n+=1
                counts[row['partition']]=counts.get(row['partition'],0)+1
                languages[row['language']]=languages.get(row['language'],0)+1
    tmp.replace(out)
    report={'source_worlds':worlds,'decision_rows':n,'renderings_per_world':60,'by_split':counts,'by_language':languages,
            'seed':seed,'fewshot':fewshot,'sha256':sha256(out),'synthetic':True,
            'scope':'Known generator templates and oracle-defined tasks. Not real-world accuracy or independent 60x data.'}
    atomic_json(out.with_suffix(out.suffix+'.manifest.json'),report);return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--worlds',type=int,required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--fewshot',type=int,choices=[0,1,4],default=0)
    a=p.parse_args(argv);print(json.dumps(generate(a.out,a.worlds,a.seed,a.fewshot),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
