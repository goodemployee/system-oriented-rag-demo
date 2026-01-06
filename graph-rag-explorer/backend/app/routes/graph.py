from fastapi import APIRouter, File, Path, Query, Request, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from pathlib import Path

router = APIRouter()

def debug_find_path(obj, prefix="root"):
    if isinstance(obj, Path):
        print(f"❌ Path found at {prefix}: {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            debug_find_path(v, f"{prefix}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            debug_find_path(v, f"{prefix}[{i}]")

@router.post("/extract_graph")
async def extract_graph(
    request: Request,
    file: UploadFile = File(...),
    max_chunks: int = 8,
):
    # 宣告需要的usecase(甚至service)
    storage = request.app.state.file_storage_service
    usecase = request.app.state.extract_graph_usecase
    
    # 將收到的檔案存檔, 並取得檔案路徑
    file_path = await storage.save(file)

    # 開始流程
    result = usecase.execute(
        file_path=file_path,
        max_chunks=max_chunks,
    )
    debug_find_path(result)
    # 回傳結果
    return JSONResponse(result)

# 知識圖譜 查詢器 單結點
@router.get("/graph")
def get_graph(
    request: Request,
    node: str = Query(..., description="節點名稱"),
) -> dict[str, object]:
    service = request.app.state.graph_query_service
    relations = service.get_related(node)
    return {"node": node, "relations": relations}


# 知識圖譜 展示 純文字
@router.get("/graph/visual")
def visual_graph(request: Request):
    service = request.app.state.graph_query_service
    data = service.get_visual_elements()
    return JSONResponse(data)


# 知識圖譜 展示 視覺化網頁
@router.get("/graph/visual/html")
def visual_graph_html():
    html_path = Path(__file__).parent / "../web/graph_visual.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
