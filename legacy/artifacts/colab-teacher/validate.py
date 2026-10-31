from datetime import datetime, UTC
import json
from pathlib import Path
import time

import httpx
from teacher_api.settings import Settings

root = Path('artifacts/colab-teacher')
base = 'http://127.0.0.1:18010'
settings = Settings()
with httpx.Client(timeout=300) as client:
	for attempt in range(30):
		try:
			response = client.get(base + '/healthz')
			if response.status_code == 200 and response.json().get('status') == 'ready':
				break
		except httpx.HTTPError:
			pass
		time.sleep(1)
	else:
		raise RuntimeError('Local teacher API did not become ready')
	unauthenticated = client.post(base + '/v1/teacher/generate', json={'messages': [{'role': 'user', 'content': 'hello'}]})
	assert unauthenticated.status_code == 401
	samples = []
	for prompt, max_tokens in [('What is 7 * 8? Reply with only the number.', 32), ('Explain in two sentences why four-bit model weights use less GPU memory than sixteen-bit weights.', 128)]:
		start = time.monotonic()
		response = client.post(base + '/v1/teacher/generate', headers={'Authorization': 'Bearer ' + settings.api_token}, json={'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0, 'max_output_tokens': max_tokens, 'enable_thinking': False})
		response.raise_for_status()
		body = response.json()
		assert body['model'] == 'Qwen3.5-9B-Q4_K_M', body['model']
		assert body['provider'] == 'colab-llamacpp-q4_k_m', body['provider']
		assert body['message']['content'].strip()
		assert '<think>' not in body['message']['content']
		if max_tokens == 32:
			assert body['message']['content'].strip() == '56', body['message']
		sample = {'prompt': prompt, 'elapsed_seconds': round(time.monotonic() - start, 3), 'response': body}
		samples.append(sample)
		print(json.dumps(sample), flush=True)
	report = {'checked_at': datetime.now(UTC).isoformat(), 'teacher_url': base, 'unauthenticated_status': unauthenticated.status_code, 'samples': samples}
	(root / 'validation.json').write_text(json.dumps(report, indent='\t') + '\n')
print('Authenticated teacher generation verified')
