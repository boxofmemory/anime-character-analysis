from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrainingOptionItem(BaseModel):
    key: str
    label: str
    path: str


class TrainingPathPreset(BaseModel):
    label: str
    path: str
    kind: str
    exists: bool = True


class TrainingTaskOption(BaseModel):
    task_type: str
    title: str
    description: str
    domain: str
    config_path: str
    default_config: dict[str, Any]
    dataset_field: str
    primary_resource_field: str | None = None
    secondary_resource_field: str | None = None
    dataset_presets: list[TrainingPathPreset] = Field(default_factory=list)
    primary_resource_options: list[TrainingOptionItem] = Field(default_factory=list)
    secondary_resource_options: list[TrainingOptionItem] = Field(default_factory=list)
    recommended_fields: list[str] = Field(default_factory=list)
    current_production: str | None = None
    current_production_experiment_id: str | None = None


class TrainingOptionsResponse(BaseModel):
    tasks: list[TrainingTaskOption]


class TrainingJobCreateRequest(BaseModel):
    task_type: str
    display_name: str | None = None
    dataset_path: str
    base_model_or_tracker_config: str | None = None
    secondary_resource_path: str | None = None
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class TrainingArtifactItem(BaseModel):
    name: str
    relative_path: str
    kind: str
    size_bytes: int | None = None
    updated_at: str | None = None
    url: str


class TrainingMetricItem(BaseModel):
    label: str
    value: str


class TrainingJobSummary(BaseModel):
    job_id: str
    task_type: str
    title: str
    status: str
    created_at: str | None = None
    updated_at: str | None = None
    experiment_dir: str
    display_name: str | None = None
    dataset_path: str | None = None
    primary_resource_path: str | None = None
    secondary_resource_path: str | None = None
    current_production: bool = False
    summary_metrics: list[TrainingMetricItem] = Field(default_factory=list)


class TrainingJobDetail(TrainingJobSummary):
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[TrainingArtifactItem] = Field(default_factory=list)
    summary_markdown: str | None = None
    log_size_bytes: int | None = None


class TrainingJobListResponse(BaseModel):
    jobs: list[TrainingJobSummary]


class TrainingLogResponse(BaseModel):
    job_id: str
    status: str
    text: str


class TrainingActionResponse(BaseModel):
    message: str
    job: TrainingJobDetail
