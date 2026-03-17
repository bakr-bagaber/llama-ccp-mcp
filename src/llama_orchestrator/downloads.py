"""Model download helpers used by the MCP control plane."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote
from urllib.request import urlretrieve

from .models import BaseModelDefinition, SourceType


class DownloadError(RuntimeError):
    """Raised when a model artifact cannot be downloaded."""


def resolve_download_url(model: BaseModelDefinition) -> str:
    """Resolve the remote URL for a model definition."""
    if model.source is SourceType.URL:
        if not model.metadata.get("url"):
            raise DownloadError(f"Model '{model.id}' is missing metadata.url for direct downloads.")
        return str(model.metadata["url"])
    if model.source is SourceType.HUGGING_FACE:
        if not model.hf_repo or not model.hf_filename:
            raise DownloadError(f"Model '{model.id}' is missing hf_repo or hf_filename.")
        repo = quote(model.hf_repo, safe="/")
        filename = quote(model.hf_filename, safe="/")
        return f"https://huggingface.co/{repo}/resolve/main/{filename}"
    raise DownloadError(f"Model '{model.id}' is not configured for remote download.")


def download_model(model: BaseModelDefinition, destination_dir: Path) -> Path:
    """Download a model artifact and return the local path."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    url = resolve_download_url(model)
    filename = model.hf_filename or Path(url).name or f"{model.id}.gguf"
    destination = destination_dir / filename
    urlretrieve(url, destination)
    return destination
