"""Append-only prospective research journal. Observations and intents are not fills."""
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def timestamp(value):
    t=datetime.fromisoformat(value.replace('Z','+00:00'))
    if t.tzinfo is None:raise ValueError('Timezone required')
    return t.astimezone(timezone.utc)


class PaperJournal:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connection() as c:
            c.execute('CREATE TABLE IF NOT EXISTS journal(seq INTEGER PRIMARY KEY, identity TEXT UNIQUE NOT NULL, recorded_at TEXT NOT NULL, payload TEXT NOT NULL, previous_hash TEXT NOT NULL, hash TEXT NOT NULL)')
    @contextmanager
    def connection(self):
        c=sqlite3.connect(self.path)
        try:
            with c:yield c
        finally:c.close()
    def register(self,strategy,protocol_hash,now=None):
        now=now or datetime.now(timezone.utc).isoformat()
        return self.append('registration',{'strategy':strategy,'protocol_hash':protocol_hash},now,identity='registration')
    def append(self,kind,payload,recorded_at,identity):
        if kind not in ('registration','observation','intent','skip'):
            raise ValueError('This journal cannot assert execution fills')
        instant=timestamp(recorded_at)
        with self.connection() as c:
            c.execute('BEGIN IMMEDIATE')
            first=c.execute('SELECT recorded_at FROM journal ORDER BY seq LIMIT 1').fetchone()
            last=c.execute('SELECT recorded_at,hash FROM journal ORDER BY seq DESC LIMIT 1').fetchone()
            if kind=='registration' and first is not None:raise ValueError('Protocol already registered')
            if kind!='registration' and first is None:raise ValueError('Register a frozen protocol first')
            if last and instant<timestamp(last[0]):raise ValueError('Cannot backdate research records')
            if kind=='observation':
                session=timestamp(payload['session_completed_at'])
                if session<=timestamp(first[0]) or session>instant:
                    raise ValueError('Observation must be completed after registration and before capture')
            if kind=='intent':
                if timestamp(payload['not_before'])<=instant:
                    raise ValueError('Execution intent must point to a future session')
            body=json.dumps({'kind':kind,'payload':payload},sort_keys=True,allow_nan=False)
            previous=last[1] if last else ''
            h=hashlib.sha256((previous+'|'+recorded_at+'|'+identity+'|'+body).encode()).hexdigest()
            c.execute('INSERT INTO journal(identity,recorded_at,payload,previous_hash,hash) VALUES(?,?,?,?,?)',(identity,recorded_at,body,previous,h))
        return h
    def verify(self):
        previous=''
        with self.connection() as c:
            for identity,at,body,prior,h in c.execute('SELECT identity,recorded_at,payload,previous_hash,hash FROM journal ORDER BY seq'):
                expected=hashlib.sha256((previous+'|'+at+'|'+identity+'|'+body).encode()).hexdigest()
                if prior!=previous or expected!=h:return False
                previous=h
        return True
