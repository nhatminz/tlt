#!/usr/bin/env python3
"""Serialize shell strings/lists as explicitly typed Hydra override values."""
import argparse,json


def encode(value,kind):
    if kind=='string':return json.dumps(value,ensure_ascii=False)
    parts=[part.strip() for part in value.split(',')] if value else []
    if any(not part for part in parts):raise ValueError('empty CSV element')
    return json.dumps(parts if kind=='strings' else [int(part) for part in parts],ensure_ascii=False,separators=(',',':'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=['string','strings','integers']);p.add_argument('value')
    a=p.parse_args();print(encode(a.value,a.kind))
