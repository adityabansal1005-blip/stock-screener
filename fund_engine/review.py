"""Evidence contract for optional bull/bear AI memos. Never a trade authorizer."""
from datetime import datetime, timezone


def utc(value):
    parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    if parsed.tzinfo is None:raise ValueError('Evidence timestamps must include timezone')
    return parsed.astimezone(timezone.utc)


def assess_memo(memo, asof):
    """Both sides cite the same point-in-time evidence pack.

    This validates structure/availability only, not truth of an AI claim or source.
    No numerical conviction is fabricated from agreement between language models.
    """
    cutoff=utc(asof);issues=[]
    evidence={}
    for item in memo.get('evidence',[]):
        identity=item.get('id')
        if not identity or identity in evidence:
            issues.append('missing_or_duplicate_evidence_id');continue
        try:
            if utc(item['available_at'])>cutoff:issues.append('future_evidence:'+identity)
        except (KeyError,ValueError,TypeError):issues.append('invalid_evidence_timestamp:'+identity)
        if not str(item.get('source_url','')).startswith('https://'):
            issues.append('missing_source:'+identity)
        evidence[identity]=item
    for side in ('bull','bear'):
        claims=memo.get(side,[])
        if not claims:issues.append('missing_'+side+'_case')
        for claim in claims:
            ids=claim.get('evidence_ids',[])
            if not claim.get('claim') or not ids or any(i not in evidence for i in ids):
                issues.append('unsupported_'+side+'_claim')
            if not claim.get('invalidation_condition'):
                issues.append('missing_'+side+'_invalidation')
    return {'memo_structurally_complete':not issues,'issues':issues,
            'human_source_review_required':True,'actionable':False,
            'role':'research_memo_only','reason':'Debate cannot override data, strategy-validation or portfolio-risk gates.'}
