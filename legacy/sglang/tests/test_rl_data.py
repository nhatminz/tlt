"""Execute the actual pinned dispatcher and its real scoring modules, no stubs.

Extract only the dispatch function to avoid importing unrelated VERL TensorDict/
Ray at package __init__ time. Relative imports still load the real reward files.
"""
import ast
import json
from pathlib import Path
import sys
import types
import pytest
from tlt_reflex.data import prepare_records,load_rows,resolve_data_source

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def dispatcher(monkeypatch):
    source=ROOT/'upstream/fastrl/verl/utils/reward_score'
    pkg=types.ModuleType('_tlt_actual_reward');pkg.__path__=[str(source)]
    monkeypatch.setitem(sys.modules,pkg.__name__,pkg)
    fn=next(n for n in ast.parse((source/'__init__.py').read_text()).body
            if isinstance(n,ast.FunctionDef) and n.name=='default_compute_score')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source/'__init__.py'),'exec'),pkg.__dict__)
    return pkg.default_compute_score


@pytest.mark.parametrize('family,source,truth,solution',[
    ('dapo','math_dapo','42',r'\boxed{42}'+'\nAnswer: 42'),
    ('gsm8k','openai/gsm8k','reasoning\n#### 42','#### 42'),
    ('math','lighteval/MATH',r'reasoning \boxed{42}',r'\boxed{42}'),
    ('simplelr','lighteval/MATH','42',r'\boxed{42}'),
])
def test_converted_dataset_dispatches_and_correct_answer_is_rewarded(dispatcher,family,source,truth,solution):
    records=prepare_records([dict(question='6*7?',answer=truth)],dataset=family)
    r=records[0]
    assert r['data_source']==source
    score=dispatcher(r['data_source'],solution,r['reward_model']['ground_truth'])
    assert (score.get('score',score.get('acc')) if isinstance(score,dict) else score)>0


@pytest.mark.parametrize('original',['math_dapo','openai/gsm8k','lighteval/MATH','DigitalLearningGmbH/MATH-lighteval','HuggingFaceH4/MATH-500','aime2024'])
def test_valid_original_identifier_is_preserved_and_reaches_real_dispatcher(dispatcher,original):
    record=prepare_records([dict(question='6*7?',answer='42',data_source=original)],dataset='dapo')[0]
    assert record['data_source']==original
    solution='#### 42' if original=='openai/gsm8k' else r'\boxed{42}'+'\nAnswer: 42'
    result=dispatcher(original,solution,record['reward_model']['ground_truth'])
    assert result is not None


def test_unknown_family_fails_and_invalid_legacy_math_is_not_accepted():
    with pytest.raises(ValueError):resolve_data_source('math',path='unknown.parquet')
    with pytest.raises(ValueError):resolve_data_source(override='math')
    assert resolve_data_source('math',path='/data/DAPO-Math-17k/train.parquet')=='math_dapo'


def test_loading_preserves_reward_metadata_and_uses_user_message(tmp_path):
    path=tmp_path/'data.json'
    path.write_text(json.dumps([dict(prompt=[dict(role='system',content='not the question'),dict(role='user',content='6*7?')],
        answer='WRONG',data_source='math_dapo',reward_model=dict(ground_truth='42'))]))
    assert load_rows(path)[0]['question']=='6*7?'
    assert load_rows(path)[0]['answer']=='42'
    assert load_rows(path)[0]['data_source']=='math_dapo'
