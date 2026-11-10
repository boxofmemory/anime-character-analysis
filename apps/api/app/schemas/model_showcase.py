from __future__ import annotations

from pydantic import BaseModel, Field


class ModelShowcaseStat(BaseModel):
    label: str
    value: str


class ModelShowcaseMetricSection(BaseModel):
    title: str
    stats: list[ModelShowcaseStat] = Field(default_factory=list)


class ModelShowcaseChartSeries(BaseModel):
    name: str
    values: list[float | None] = Field(default_factory=list)


class ModelShowcaseChart(BaseModel):
    key: str
    title: str
    chart_type: str = "line"
    summary: str | None = None
    x_labels: list[str] = Field(default_factory=list)
    series: list[ModelShowcaseChartSeries] = Field(default_factory=list)
    y_axis_label: str | None = None


class ModelShowcaseVisualItem(BaseModel):
    title: str
    image_url: str
    caption: str | None = None


class ModelShowcaseVisualGroup(BaseModel):
    title: str
    description: str | None = None
    items: list[ModelShowcaseVisualItem] = Field(default_factory=list)


class ModelShowcaseCard(BaseModel):
    key: str
    title: str
    artifact_name: str
    summary: str
    selection_badge: str = "CURRENT SELECTION"
    selection_hint: str | None = None
    selected_experiment_id: str | None = None
    capabilities: list[str]
    parameters: list[ModelShowcaseStat]
    datasets: list[ModelShowcaseStat]
    metrics: list[ModelShowcaseStat]
    metric_sections: list[ModelShowcaseMetricSection] = Field(default_factory=list)
    charts: list[ModelShowcaseChart] = Field(default_factory=list)
    visual_groups: list[ModelShowcaseVisualGroup] = Field(default_factory=list)
    current_production_experiment_id: str | None = None


class ModelShowcaseExperimentArtifact(BaseModel):
    name: str
    kind: str
    url: str


class ModelShowcaseExperiment(BaseModel):
    job_id: str
    task_type: str
    title: str
    status: str
    created_at: str | None = None
    updated_at: str | None = None
    current_production: bool = False
    selected: bool = False
    summary_metrics: list[ModelShowcaseStat]
    artifacts: list[ModelShowcaseExperimentArtifact]


class ModelShowcaseGroup(BaseModel):
    task_type: str
    title: str
    current_production_experiment_id: str | None = None
    experiments: list[ModelShowcaseExperiment]


class ModelShowcaseResponse(BaseModel):
    cards: list[ModelShowcaseCard]
    experiment_groups: list[ModelShowcaseGroup]
