"""Serial local candidate generation; real DSPy BestOfN selection. No judge LM."""
import json, math, time, urllib.request, sys, subprocess, fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PYTHON='/home/mitch/venvs/crossword-batch/bin/python'
def payload(q):
 return {k:q[k] for k in ('grid','clues','title') if k in q}
def valid(q,p):
 g=p.get('grid'); src=q['grid']
 return isinstance(g,list) and len(g)==len(src) and all(isinstance(a,str) and len(a)==len(b) and all((x=='#' if y=='#' else x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' and (y=='.' or x==y)) for x,y in zip(a,b)) for a,b in zip(g,src))
def select(q,pool,ranker='confidence'):
 import dspy
 if ranker not in ('confidence','consensus'):raise ValueError('unknown ranker')
 good=[i for i,p in enumerate(pool) if valid(q,p)]
 if not good:raise ValueError('no valid candidates')
 cells=[(r,c) for r,row in enumerate(q['grid']) for c,x in enumerate(row) if x=='.']
 scores=[]
 for i in good:
  if ranker=='confidence':
   vals=[float(pool[i].get('confidence',{}).get(f'{r},{c}',0)) for r,c in cells]
   score=sum(v if math.isfinite(v) and 0<=v<=1 else 0 for v in vals)/max(1,len(vals))
  else:score=sum(pool[i]['grid'][r][c]==pool[j]['grid'][r][c] for r,c in cells for j in good)/max(1,len(cells)*len(good))
  scores.append(score)
 # BestOfN deepcopies modules; mutable iterator intentionally lives in closure.
 stream=iter(zip(good,scores))
 class Pool(dspy.Module):
  def get_lm(self):return None
  def forward(self,**kwargs):
   i,s=next(stream);return dspy.Prediction(index=i,reward=s)
 # BestOfN requires an LM config to copy rollout IDs, even with no predictors.
 # This LM is NEVER called; all predictions above are pre-generated locally.
 with dspy.context(lm=dspy.LM('openai/unused-local-pool',api_key='unused')):
  pred=dspy.BestOfN(Pool(),N=len(good),reward_fn=lambda args,p:p.reward,threshold=float('inf'))()
 return dict(engine='dspy.BestOfN',version=dspy.__version__,ranker=ranker,selected_index=pred.index,scores=dict(zip(good,scores)),rejected=[i for i in range(len(pool)) if i not in good],seed_supported=False,warning='Heuristic selection, not a trained judge. Confidence is reveal-time, temperature-scaled probability; not calibrated across temperatures.')
def generate(q,n=4,temperature=.7,steps=12,remask_frac=0,fetch=None,url='http://127.0.0.1:8011'):
 if type(n)!=int or not 1<=n<=4:raise ValueError('n must be 1..4')
 if steps not in (12,24) or temperature not in (.4,.7) or remask_frac not in (0,.1):raise ValueError('unsupported settings')
 q=payload(q)
 if not q.get('grid') or not all(isinstance(r,str) and len(r)==len(q['grid'][0]) and set(r)<=set('.#ABCDEFGHIJKLMNOPQRSTUVWXYZ') for r in q['grid']):raise ValueError('invalid input grid')
 def http(req):
  with urllib.request.urlopen(urllib.request.Request(url+'/fill',data=json.dumps(req).encode(),headers={'Content-Type':'application/json'}),timeout=45) as f:return json.load(f)
 fetch=fetch or http;pool=[];requests=[];errors=[];start=time.monotonic()
 for i in range(n):
  req=dict(q,steps=12 if i==0 else steps,temperature=0 if i==0 else temperature,samples=1,remask_frac=0 if i==0 else remask_frac,snapshots=True)
  requests.append(req)
  try:
   p=fetch(req)
   if not valid(q,p):raise ValueError('invalid candidate')
   pool.append(p)
  except Exception as e:
   errors.append(repr(e))
   if not pool:raise
   break # Never overlap another request after timeout.
 return pool,dict(requests=requests,errors=errors,fallback=bool(errors),elapsed_s=time.monotonic()-start,serial=True,seed_supported=False)
def solve(q,url='http://127.0.0.1:8011'):
 # Cross-process lock also shared by evaluator; bounded wait by nonblocking acquire.
 with open('/tmp/crossword-batch.lock','w') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('batch evaluator/solve busy; retry later')
  settings_file=ROOT/'assessment/batch-selection/frozen.json'
  settings=json.loads(settings_file.read_text()) if settings_file.exists() else dict(temperature=.7,steps=12,remask_frac=0,ranker='consensus')
  pool,meta=generate(q,url=url,**{k:settings[k] for k in ('temperature','steps','remask_frac')})
  try:
   proc=subprocess.run([PYTHON,'-m','crossword_agent.batch_selection'],input=json.dumps(dict(q=payload(q),pool=pool,ranker=settings['ranker'])),text=True,capture_output=True,timeout=30,cwd=ROOT,check=True)
   selection=json.loads(proc.stdout)
  except Exception as e:selection=dict(selected_index=0,engine='baseline-fallback',error=repr(e));meta['fallback']=True
  result=dict(pool[selection['selected_index']]);result['selector']={**meta,**selection,'candidate_count':len(pool)};return result
if __name__=='__main__':
 x=json.load(sys.stdin);print(json.dumps(select(**x)))
