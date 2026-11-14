import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

root = Path('/content/capstone-teacher')
manifest = json.loads((root / 'manifest.json').read_text())
models = root / 'models'
models.mkdir(exist_ok=True)
runtime = root / 'runtime'
runtime.mkdir(exist_ok=True)
model = manifest['model']

def download(url, destination, digest):
	if destination.exists():
		with destination.open('rb') as stream:
			actual = hashlib.file_digest(stream, 'sha256').hexdigest()
		if actual == digest:
			print('Verified cached', destination.name, flush=True)
			return destination
	partial = destination.with_suffix(destination.suffix + '.partial')
	print('Downloading', destination.name, flush=True)
	subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--retry', '5', '--retry-delay', '3', '--output', str(partial), url], check=True)
	with partial.open('rb') as stream:
		actual = hashlib.file_digest(stream, 'sha256').hexdigest()
	if actual != digest:
		raise RuntimeError('Checksum mismatch for ' + destination.name)
	partial.replace(destination)
	print('Downloaded and verified', destination.name, destination.stat().st_size, flush=True)
	return destination

jobs = [(a['url'], root / a['name'], a['digest'].removeprefix('sha256:')) for a in manifest['assets']]
jobs.append((f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{model['file']}?download=true", models / model['file'], model['sha256']))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
	list(pool.map(lambda job: download(*job), jobs))
for asset in manifest['assets']:
	with tarfile.open(root / asset['name']) as archive:
		archive.extractall(runtime, filter='data')
servers = list(runtime.rglob('llama-server'))
if len(servers) != 1:
	raise RuntimeError('Expected one llama-server executable')
server = servers[0]
libraries = {str(p.parent) for p in runtime.rglob('*.so*')}
os.environ['LD_LIBRARY_PATH'] = ':'.join(sorted(libraries)) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
subprocess.run([str(server), '--version'], check=True)
command = [str(server), '--model', str(models / model['file']), '--alias', 'Qwen3.5-9B-Q8_0', '--host', '127.0.0.1', '--port', '8001', '--n-gpu-layers', '99', '--ctx-size', '16384', '--parallel', '4', '--jinja', '--reasoning', 'off', '--reasoning-format', 'deepseek']
print('Starting pinned quantized teacher', flush=True)
os.execv(str(server), command)
