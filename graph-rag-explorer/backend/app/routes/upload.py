from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
from app.config.paths import UPLOAD_DIR
from fastapi import APIRouter, Query, Request
from app.application.usecases.upload_usecase import UploadUseCase

router = APIRouter()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    storage = request.app.state.file_storage_service
    usecase = request.app.state.upload_usecase
    
    file_path = await storage.save(file)
    return await usecase.execute(file_path)