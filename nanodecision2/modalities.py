"""Explicit records/table/time-series input contracts (stdlib only).

No files/URLs are opened from a request. Numeric transforms are fixed signed-log
features, not statistics fitted on validation/test or per-window normalization.
Original values and units remain in field descriptions.
"""
import json
import math

_MISSING=object()


def number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)):
        raise ValueError('Expected a numeric observation')
    try:result=float(value)
    except OverflowError:raise ValueError('Numeric observation exceeds finite float range') from None
    if not math.isfinite(result):raise ValueError('Non-finite observed numeric value')
    return result


def signed_log(value):
    v=number(value)
    return math.copysign(math.log1p(abs(v)),v)


def field_features(value=_MISSING):
    # missing/null/bool/number/text and numeric sign/log/zero. 0 != null != absent.
    out=[0.]*8
    if value is _MISSING:out[0]=1.
    elif value is None:out[1]=1.
    elif isinstance(value,bool):out[2]=1.;out[5]=float(value)
    elif isinstance(value,(int,float)):
        v=number(value);out[3]=1.;out[5]=signed_log(v);out[6]=float(v<0);out[7]=float(v==0)
    elif isinstance(value,str):out[4]=1.
    else:raise ValueError('Structured cells must be scalar; flatten nested objects explicitly')
    return out


def structured_tokens(state,max_cells=512):
    if not isinstance(state,dict) or state.get('modality') not in ('records','table'):
        raise ValueError('Expected records/table input')
    if set(state)-{'modality','fields','columns','rows','ordered_rows'}:
        raise ValueError('Unknown structured input field; no arbitrary file/URL references')
    texts=[];features=[];positions=[];groups=[]
    def add(name,value,unit,rowpos,group=0):
        if not isinstance(name,str) or not name or not isinstance(unit,str):raise ValueError('Field name/unit must be strings')
        if len(texts)>=max_cells:raise ValueError('Structured cell capacity exceeded; no truncation')
        payload='<MISSING>' if value is _MISSING else json.dumps(value,ensure_ascii=False,allow_nan=False)
        texts.append(f'FIELD: {name}\nUNIT: {unit}\nVALUE: {payload}')
        features.append(field_features(value));positions.append(float(rowpos));groups.append(group)
    if state['modality']=='records':
        fields=state.get('fields')
        if not isinstance(fields,list) or not fields:raise ValueError('Nonempty fields list required')
        names=[]
        for field in sorted(fields,key=lambda f:f.get('name','')):
            if not isinstance(field,dict) or set(field)-{'name','value','unit'}:raise ValueError('Invalid scalar field')
            name=field.get('name');names.append(name);add(name,field.get('value',_MISSING),field.get('unit',''),0)
        if len(set(names))!=len(names):raise ValueError('Duplicate field names')
    else:
        columns=state.get('columns');rows=state.get('rows');ordered=state.get('ordered_rows',False)
        if type(ordered) is not bool:raise ValueError('ordered_rows must be Boolean')
        if not isinstance(columns,list) or not columns or not isinstance(rows,list) or not rows:raise ValueError('Nonempty columns/rows required')
        names=[c.get('name') if isinstance(c,dict) else None for c in columns]
        if any(not isinstance(n,str) or not n for n in names) or len(set(names))!=len(names):raise ValueError('Invalid/duplicate column names')
        if len(columns)*len(rows)>max_cells:raise ValueError('Table capacity exceeded; no truncation')
        for ri,row in enumerate(rows):
            if not isinstance(row,dict) or set(row)-set(names):raise ValueError('Rows must map declared column names to scalar values')
            for col in sorted(columns,key=lambda c:c['name']):
                if set(col)-{'name','unit'}:raise ValueError('Invalid column metadata')
                add(col['name'],row.get(col['name'],_MISSING),col.get('unit',''),ri+1 if ordered else 0,ri)
    return {'texts':texts,'features':features,'positions':positions,'groups':groups}


def sensor_tokens(state,patch_size=16,max_frames=8192,max_channels=32,max_patches=2048):
    allowed={'modality','values','timestamps','sample_rate_hz','channel_names','units','observed_mask','quality_mask','prediction_time'}
    if not isinstance(state,dict) or state.get('modality')!='timeseries' or set(state)-allowed:
        raise ValueError('Expected inline time-series contract; external paths are not read')
    values=state.get('values');names=state.get('channel_names');units=state.get('units')
    if not isinstance(values,list) or not 1<=len(values)<=max_frames:raise ValueError('Invalid frame count')
    if not isinstance(names,list) or not 1<=len(names)<=max_channels or any(not isinstance(x,str) or not x for x in names) or len(set(names))!=len(names):raise ValueError('Invalid channel names')
    t,c=len(values),len(names)
    if not isinstance(units,list) or len(units)!=c or not all(isinstance(u,str) for u in units):raise ValueError('Explicit units required for every channel')
    if any(not isinstance(row,list) or len(row)!=c for row in values):raise ValueError('values must have rectangular [T,C] shape')
    if 'timestamps' in state and 'sample_rate_hz' in state:raise ValueError('Supply timestamps or sample rate, not both')
    if 'timestamps' in state:
        times=state['timestamps']
        if not isinstance(times,list) or len(times)!=t:raise ValueError('Timestamp length mismatch')
        times=[number(x) for x in times]
        if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Timestamps must strictly increase')
    else:
        rate=number(state.get('sample_rate_hz'))
        if rate<=0:raise ValueError('Positive sample_rate_hz required')
        times=[i/rate for i in range(t)]
    if 'prediction_time' in state and times[-1]>number(state['prediction_time']):raise ValueError('Samples after prediction_time leak future observations')
    masks=[]
    for key in ('observed_mask','quality_mask'):
        mask=state.get(key,[[True]*c for _ in range(t)])
        if not isinstance(mask,list) or len(mask)!=t or any(not isinstance(row,list) or len(row)!=c or any(type(x) is not bool for x in row) for row in mask):raise ValueError('Masks must have Boolean [T,C] shape')
        masks.append(mask)
    if patch_size<1 or math.ceil(t/patch_size)*c>max_patches:raise ValueError('Sensor patch budget exceeded')
    texts=[];patches=[];positions=[]
    # Channel IDs/units are read as text, never ordinal class indexes.
    for channel in sorted(range(c),key=lambda j:names[j]):
        for lo in range(0,t,patch_size):
            transformed=[];observed=[];deltas=[]
            for i in range(lo,lo+patch_size):
                ok=i<t and masks[0][i][channel] and masks[1][i][channel]
                transformed.append(signed_log(values[i][channel]) if ok else 0.)
                observed.append(float(ok))
                deltas.append(math.log1p(times[i]-times[i-1]) if i<t and i>0 else 0.)
            texts.append(f'CHANNEL: {names[channel]}\nUNIT: {units[channel]}')
            patches.append(transformed+observed+deltas)
            positions.append(math.log1p(times[lo]-times[0]))
    return {'texts':texts,'features':patches,'positions':positions,'frames':t,'channels':c,
            'observed_count':sum(masks[0][i][j] and masks[1][i][j] for i in range(t) for j in range(c))}
