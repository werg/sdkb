from pathlib import Path
import json
from sdkb.evaluation import evaluate_transfer_run
from sdkb.checkpoints import resolve_checkpoint
from sdkb.operations import atomic_json
from sdkb.trajectories import file_sha256

root=Path('/archive/diagnostics/binding-short-muon-20260919')
root.mkdir(parents=True, exist_ok=True)
sources={
 'warmup': Path('/runs/binding-pilot-20260919/latent_warmup'),
 'muon_100': Path('/archive/profiles/binding-muon-checkpoint-off-20260919/training'),
}
for name, source in sources.items():
 checkpoint=resolve_checkpoint(source, verify=True)
 view=root/name
 view.mkdir(exist_ok=True)
 link=view/'checkpoints'
 if not link.exists():
  link.symlink_to(checkpoint.parent.resolve(), target_is_directory=True)
 (view/'CURRENT').write_text(checkpoint.name+'\n')
 result=evaluate_transfer_run(view, '/runs/binding-pilot-20260919/data/validation.jsonl',
   drop_supports=True, binding_counterfactuals=True)
 summary={k:v for k,v in result.items() if k!='rows'}
 summary['checkpoint_manifest_sha256']=file_sha256(checkpoint/'manifest.json')
 summary['source_checkpoint']=str(checkpoint)
 atomic_json(root/(name+'.json'),summary)
 print(json.dumps({'stage':name,'by_family':result['by_family']}), flush=True)
