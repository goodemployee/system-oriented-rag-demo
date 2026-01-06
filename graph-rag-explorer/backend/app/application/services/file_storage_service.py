from pathlib import Path
from fastapi import UploadFile

from app.infrastructure.storage.local_file_storage import LocalFileStorage


class FileStorageService:
    def __init__(self, upload_dir: Path):
        self._storage = LocalFileStorage(upload_dir)

    async def save(self, file: UploadFile) -> Path:
        return await self._storage.save_upload(file)
