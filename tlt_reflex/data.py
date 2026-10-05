"""Local SpecNaacl prompt/data conventions; FastRL reward/RL remain upstream."""
def load_rows(path):
    import pandas as pd
    from pathlib import Path
    p=Path(path)
    if p.suffix=='.parquet':data=pd.read_parquet(p).to_dict(orient='records')
    elif p.suffix in ('.json','.jsonl'):data=pd.read_json(p,lines=p.suffix=='.jsonl').to_dict(orient='records')
    else:raise ValueError('configured dataset must be local parquet/json/jsonl')
    rows=[]
    for r in data:
        q=r.get('question',r.get('prompt'))
        if hasattr(q,'tolist'):q=q.tolist()
        if isinstance(q,list):
            users=[m.get('content') for m in q if isinstance(m,dict) and m.get('role')=='user']
            q=users[-1] if users else q[0]['content']
        reward=r.get('reward_model') or {}
        answer=reward.get('ground_truth',r.get('answer',r.get('solution')))
        rows.append(dict(question=q,answer=answer,data_source=r.get('data_source'),
                         extra_info=r.get('extra_info') or {}))
    return rows


def prompt_messages(question,source=None):
    # Exact instruction text taken from Source TrainDataCollator.
    user='''Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            Your response should include your thought process enclosed within <think></think> tags
            and the final answer enclosed within <answer></answer> tags (Just put a number between the tags).\n
            ### Instruction:\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}'''
    text=user.format_map({'instruction':question})
    # Match exact output formats expected by the UNCHANGED upstream rewards.
    if source=='openai/gsm8k':text+='\nAlso end with a final line: #### <numeric answer>'
    elif source=='math_dapo' or isinstance(source,str) and source.startswith('aime'):
        text+='\nAlso end with a final line: Answer: <final answer>'
    return [{'role':'system','content':'You are a math problem assistant.'},
            {'role':'user','content':text}]


# Exact math-family identifiers from the pinned upstream reward dispatcher.
MATH_SOURCES={'lighteval/MATH','DigitalLearningGmbH/MATH-lighteval','HuggingFaceH4/MATH-500'}
PRIME_SOURCES={'numina_aops_forum','numina_synthetic_math','numina_amc_aime',
               'numina_synthetic_amc','numina_cn_k12','numina_olympiads'}


def valid_math_source(value):
    return isinstance(value,str) and (value in MATH_SOURCES|PRIME_SOURCES|{'openai/gsm8k','math_dapo'} or value.startswith('aime'))


def resolve_data_source(original=None,*,dataset=None,path='',override=None):
    if override is not None:
        if not valid_math_source(override):raise ValueError('unsupported explicit math reward data_source: '+override)
        return override
    if valid_math_source(original):return original
    # Invalid legacy `math` must be mapped by actual dataset identity, not
    # blindly mapped to DAPO. Unknown families require an explicit override.
    identity=(dataset or path).lower()
    if 'gsm8k' in identity:return 'openai/gsm8k'
    if 'dapo' in identity:return 'math_dapo'
    if any(x in identity for x in ('math','simplelr','abel')):return 'lighteval/MATH'
    raise ValueError(f'cannot determine supported reward data_source for {original!r}; specify --dataset dapo/gsm8k/math or --data-source')


def normalize_ground_truth(answer,source):
    # Dispatcher GSM8K compares the raw numeric answer, MATH compares the
    # unboxed answer. Boxing *ground truth* here would silently give reward0.
    value=str(answer).strip()
    if source=='openai/gsm8k' and '####' in value:value=value.rsplit('####',1)[1].strip()
    start=value.rfind('\\boxed{')
    if start>=0:
        begin=start+len('\\boxed{');depth=1
        for i in range(begin,len(value)):
            depth+=(value[i]=='{')-(value[i]=='}')
            if depth==0:
                value=value[begin:i];break
        else:raise ValueError('malformed boxed ground-truth answer')
    if source=='openai/gsm8k':value=value.replace(',','').replace('$','')
    if not value:raise ValueError('empty ground-truth answer')
    return value


def prepare_records(rows,*,path='',dataset=None,data_source=None):
    prepared=[]
    for i,row in enumerate(rows):
        if row['answer'] is None or not isinstance(row['question'],str) or not row['question'].strip():
            raise ValueError(f'dataset row {i} lacks a real question/ground-truth answer')
        source=resolve_data_source(row.get('data_source'),dataset=dataset,path=path,override=data_source)
        truth=normalize_ground_truth(row['answer'],source)
        prepared.append(dict(data_source=source,prompt=prompt_messages(row['question'],source),ability='math',
            reward_model=dict(style='rule',ground_truth=truth),
            extra_info=dict(row.get('extra_info') or {},index=i,source_path=path)))
    return prepared
