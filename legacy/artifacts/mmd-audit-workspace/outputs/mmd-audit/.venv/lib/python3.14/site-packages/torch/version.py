from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip', 'rocm', 'xpu']
__version__ = '2.15.0+cpu'
debug = False
cuda: Optional[str] = None
git_version = 'd0f2af8b66b20da4baa50644c0cd3f887a2eae2e'
hip: Optional[str] = None
rocm: Optional[str] = None
xpu: Optional[str] = None
