"""Query encoders. The main encoder is BAAI/bge-small-en-v1.5 (paper Section III-B)."""

from typing import Any, Protocol

import numpy as np

from harness.mmd import Array

BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


class Encoder(Protocol):
	def encode(self, texts: list[str]) -> Array:
		"""Unit-length embeddings, one row per text."""
		...

	def identity(self) -> dict[str, Any]:
		"""Model, revision, and limits, recorded in the calibration artifact."""
		...


class BGEEncoder:
	"""CLS-pooled, L2-normalized BGE embeddings of the current user turn only.

	Inputs longer than max_tokens are truncated; `truncated` counts them for reporting.
	"""

	def __init__(
		self,
		model: str = BGE_MODEL,
		revision: str = BGE_REVISION,
		max_tokens: int = 512,
		batch_size: int = 64,
		device: str | None = None,
	) -> None:
		try:
			import torch
			from transformers import AutoModel, AutoTokenizer
		except ImportError as error:
			raise RuntimeError(
				"the BGE encoder needs `uv sync --all-packages --extra embed`"
			) from error
		self._torch = torch
		self.model_name, self.revision, self.max_tokens = model, revision, max_tokens
		self.batch_size = batch_size
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
		self.model = AutoModel.from_pretrained(model, revision=revision).to(self.device).eval()
		self.encoded = 0
		self.truncated = 0

	def encode(self, texts: list[str]) -> Array:
		rows: list[Array] = []
		for start in range(0, len(texts), self.batch_size):
			batch = texts[start : start + self.batch_size]
			lengths = self.tokenizer(batch, add_special_tokens=True)["input_ids"]
			self.truncated += sum(len(ids) > self.max_tokens for ids in lengths)
			inputs = self.tokenizer(
				batch,
				padding=True,
				truncation=True,
				max_length=self.max_tokens,
				return_tensors="pt",
			).to(self.device)
			with self._torch.inference_mode():
				cls = self.model(**inputs).last_hidden_state[:, 0]
				cls = self._torch.nn.functional.normalize(cls, dim=-1)
			rows.append(cls.float().cpu().numpy().astype(np.float64))
		self.encoded += len(texts)
		return np.concatenate(rows) if rows else np.zeros((0, 0))

	def identity(self) -> dict[str, Any]:
		return {
			"model": self.model_name,
			"revision": self.revision,
			"max_tokens": self.max_tokens,
			"pooling": "cls",
			"normalized": True,
			"encoded": self.encoded,
			"truncated": self.truncated,
		}
