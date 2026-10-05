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
        if isinstance(q,list):q=q[0]['content']
        answer=r.get('answer',r.get('reward_model',{}).get('ground_truth'))
        rows.append(dict(question=q,answer=answer))
    return rows


def prompt_messages(question):
    # Exact instruction text taken from Source TrainDataCollator.
    user='''Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            Your response should include your thought process enclosed within <think></think> tags
            and the final answer enclosed within <answer></answer> tags (Just put a number between the tags).\n
            ### Instruction:\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}'''
    return [{'role':'system','content':'You are a math problem assistant.'},
            {'role':'user','content':user.format_map({'instruction':question})}]
