from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()/'scripts'))
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from probe_global_routing import global_features,address_scores
from probe_routing_stop import StopProbe
from sdkb.data import load_episodes
from sdkb.store import DiskStore,StoredRecord
from sdkb.operations import atomic_json
from sdkb.trajectories import file_sha256
root=Path('/archive/profiles/routing-score-precision-20260919')
features_root=Path('/archive/probes/routing-breadth-20260919')
endpoint=Path('/archive/probes/global-negatives-long-20260919/global/resume.pt')
state=torch.load(endpoint,weights_only=True,map_location='cpu')
features=load_file(str(features_root/'heldout-features.safetensors'))
data,ids=global_features(features,load_episodes(features_root/'heldout.jsonl'))
data={k:v.to('cuda') for k,v in data.items()}
model=StopProbe(state['model'],False).cuda().eval().requires_grad_(False)
torch.set_num_threads(8)
torch.set_float32_matmul_precision('highest')
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
 keys=F.normalize(model.address(F.normalize(model.key(data['key']),dim=-1)),dim=-1).float()
 queries=model.query_map(F.normalize(model.query_head(data['raw_query']),dim=-1)).float()
 scores=address_scores(model,data,torch.arange(len(queries),device='cuda'),'global').cpu()*.1
assert scores.dtype==torch.float32
store=DiskStore(root/'bank.sqlite')
store.put_many(StoredRecord(rid,key.cpu(),torch.zeros(1),created_at=int(time),source_id=rid)
               for rid,key,time in zip(ids,keys,data['created_at'],strict=True))
errors=[];ranks=[]
for i,query in enumerate(queries):
 plan=store.search(query.cpu(),top_k=len(ids),query_time=int(data['query_time'][i]))
 expected={s.record_id:s.score for s in plan.selections}
 errors.extend(abs(float(scores[i,j])-expected[rid]) for j,rid in enumerate(ids))
 ranks.append([ids[j] for j in scores[i].argsort(descending=True,stable=True)[:2]]==[s.record_id for s in plan.selections[:2]])
assert max(errors)<2e-6 and all(ranks)
atomic_json(root/'result.json',{'source_commit':'148933e','endpoint_sha256':file_sha256(endpoint),
 'feature_sha256':file_sha256(features_root/'heldout-features.safetensors'),
 'device':'native CUDA BF16 projection / FP32 exact cosine','queries':len(queries),'records':len(ids),
 'max_cosine_abs_error_vs_diskstore':max(errors),'identical_top_two_rankings':sum(ranks),
 'driver_sha256':file_sha256(__file__),
 'notice':'Shared batched projected vectors serialized/reloaded. Does not assert single-writer versus batched GEMM bit identity.'})
print({'max_error':max(errors),'identical_rankings':sum(ranks)},flush=True)
