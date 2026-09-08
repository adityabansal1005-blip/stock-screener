"""Read-only research status. No subprocesses, external APIs or order endpoints."""
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from .review import assess_memo

router=APIRouter()
ROOT=Path(__file__).resolve().parents[1]


class MemoRequest(BaseModel):
    memo: dict
    asof: str


@router.post('/api/fund-research/review')
def review(request: MemoRequest):
    try:
        return assess_memo(request.memo,request.asof)
    except (ValueError,KeyError,TypeError,AttributeError):
        raise HTTPException(status_code=422,detail='Malformed evidence memo or timestamp')


@router.get('/api/fund-research')
def status():
    runs=ROOT/'fund_runs'
    files=sorted(runs.glob('*/results.json'),key=lambda p:p.stat().st_mtime,reverse=True) if runs.exists() else []
    if not files:
        return {'state':'not_run','mode':'research_only','promotion':'not_approved','strategies':{},
                'audit':{'blockers':['No local fund research run. Use RESEARCH.cmd after obtaining the historical archive.']}}
    try:
        if files[0].stat().st_size>2000000:raise ValueError('Oversized result')
        result=json.loads(files[0].read_text())
        return {**result,'state':'available','run':files[0].parent.name}
    except (OSError,ValueError):
        return {'state':'unreadable','promotion':'not_approved','strategies':{},'audit':{'blockers':['Local research result could not be read.']}}


@router.get('/fund-research',response_class=HTMLResponse)
def page():
    return (ROOT/'templates/fund_research.html').read_text(encoding='utf-8')
