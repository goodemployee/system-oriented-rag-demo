from pathlib import Path
from fastapi import UploadFile

### 職責：只負責把檔案存成實體檔案
class LocalFileStorage:
    def __init__(self, upload_dir: Path):
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, file: UploadFile) -> Path:
        """
        將 UploadFile 儲存到本地並回傳檔案路徑
        """
        file_path = self.upload_dir / file.filename  # type: ignore

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        return file_path
