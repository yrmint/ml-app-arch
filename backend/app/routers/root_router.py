from fastapi import APIRouter

router = APIRouter(tags=["Root"])


@router.get("/")
def read_root():
    return {"message": "Genre classification"}
