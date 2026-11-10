from pydantic import BaseModel, Field


class RenameStrangerRequest(BaseModel):
    old_name: str = Field(..., min_length=1)
    new_name: str = Field(..., min_length=1)
    recognition_model_path: str | None = None


class RenameStrangerResponse(BaseModel):
    status: str
    message: str
    created: bool
    fused: bool
