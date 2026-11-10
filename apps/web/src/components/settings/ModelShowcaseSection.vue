<template>
  <section class="section-card showcase-section">
    <div class="section-header">
      <div>
        <div class="eyebrow">Model Showcase</div>
        <h2>模型展示</h2>
        <p>按当前模型组合展示训练指标、评估图表、版本历史与可视化成果。</p>
      </div>
      <el-button type="primary" plain @click="emit('refresh')">刷新结果</el-button>
    </div>

    <div class="selection-panel">
      <div class="selection-card">
        <span>检测模型</span>
        <el-select v-model="selectedDetectionModel" class="selection-select" @change="emitModelSelection">
          <el-option
            v-for="item in props.analysisModelOptions?.detection?.options || []"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
      </div>

      <div class="selection-card">
        <span>特征模型</span>
        <el-select v-model="selectedRecognitionModel" class="selection-select" @change="emitModelSelection">
          <el-option
            v-for="item in props.analysisModelOptions?.recognition?.options || []"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
      </div>

      <div class="selection-card">
        <span>跟踪配置</span>
        <el-select v-model="selectedTrackerConfig" class="selection-select" @change="emitModelSelection">
          <el-option
            v-for="item in props.analysisModelOptions?.tracker?.options || []"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
      </div>
    </div>

    <div class="showcase-container">
      <el-tabs v-model="activeTab" class="showcase-tabs">
        <el-tab-pane
          v-for="tab in showcaseTabs"
          :key="tab.key"
          :label="tab.title"
          :name="tab.key"
        >
          <div v-if="tab.card" class="tab-content">
            <article class="production-crown">
              <div class="showcase-top">
                <div>
                  <div class="crown-badge">{{ tab.card.selection_badge || 'CURRENT SELECTION' }}</div>
                  <div class="showcase-label">{{ tab.card.title }}</div>
                  <strong class="showcase-artifact">{{ tab.card.artifact_name }}</strong>
                  <p v-if="tab.card.selection_hint" class="selection-hint">{{ tab.card.selection_hint }}</p>
                </div>
                <div class="showcase-side">
                  <span class="showcase-key">{{ tab.card.key.toUpperCase() }}</span>
                  <span v-if="tab.card.selected_experiment_id" class="production-pill">
                    {{ tab.card.selected_experiment_id }}
                  </span>
                </div>
              </div>

              <p class="showcase-summary">{{ tab.card.summary }}</p>

              <div class="showcase-block">
                <div class="showcase-block-title">核心能力</div>
                <ul class="showcase-list">
                  <li v-for="item in tab.card.capabilities" :key="item">{{ item }}</li>
                </ul>
              </div>

              <div class="showcase-stat-grid">
                <div class="showcase-stat-box">
                  <div class="showcase-block-title">训练参数</div>
                  <div class="showcase-stat-list">
                    <div v-for="item in tab.card.parameters" :key="`${tab.card.key}-${item.label}`" class="showcase-stat-item">
                      <span>{{ item.label }}</span>
                      <strong>{{ item.value }}</strong>
                    </div>
                  </div>
                </div>

                <div class="showcase-stat-box">
                  <div class="showcase-block-title">数据集 / 资源</div>
                  <div class="showcase-stat-list">
                    <div v-for="item in tab.card.datasets" :key="`${tab.card.key}-dataset-${item.label}`" class="showcase-stat-item">
                      <span>{{ item.label }}</span>
                      <strong>{{ item.value }}</strong>
                    </div>
                  </div>
                </div>

                <div class="showcase-stat-box showcase-highlight">
                  <div class="showcase-block-title">核心检验指标</div>
                  <div class="showcase-stat-list">
                    <div v-for="item in tab.card.metrics" :key="`${tab.card.key}-metric-${item.label}`" class="showcase-stat-item">
                      <span>{{ item.label }}</span>
                      <strong>{{ item.value }}</strong>
                    </div>
                  </div>
                </div>
              </div>

              <div v-if="(tab.card.metric_sections || []).length > 0" class="metric-section-grid">
                <div
                  v-for="section in tab.card.metric_sections"
                  :key="`${tab.card.key}-metric-section-${section.title}`"
                  class="showcase-stat-box"
                >
                  <div class="showcase-block-title">{{ section.title }}</div>
                  <div class="showcase-stat-list">
                    <div
                      v-for="item in section.stats"
                      :key="`${tab.card.key}-${section.title}-${item.label}`"
                      class="showcase-stat-item"
                    >
                      <span>{{ item.label }}</span>
                      <strong>{{ item.value }}</strong>
                    </div>
                  </div>
                </div>
              </div>

              <div v-if="(tab.card.charts || []).length > 0" class="chart-gallery">
                <MetricChartPanel
                  v-for="chart in tab.card.charts"
                  :key="`${tab.card.key}-${chart.key}`"
                  :chart="chart"
                />
              </div>

              <div v-if="(tab.card.visual_groups || []).length > 0" class="visual-block">
                <div
                  v-for="group in tab.card.visual_groups"
                  :key="`${tab.card.key}-visual-${group.title}`"
                  class="visual-group"
                >
                  <div class="visual-head">
                    <div class="showcase-block-title">{{ group.title }}</div>
                    <div v-if="group.description" class="visual-desc">{{ group.description }}</div>
                  </div>
                  <div class="visual-grid">
                    <a
                      v-for="item in group.items"
                      :key="`${group.title}-${item.image_url}`"
                      :href="toApiUrl(item.image_url)"
                      target="_blank"
                      rel="noreferrer"
                      class="visual-card"
                    >
                      <img :src="toApiUrl(item.image_url)" :alt="item.title" />
                      <strong>{{ item.title }}</strong>
                      <span v-if="item.caption">{{ item.caption }}</span>
                    </a>
                  </div>
                </div>
              </div>
            </article>

            <div v-if="tab.group" class="history-block">
              <div class="history-header">
                <div class="history-title">版本演进历史</div>
                <div class="history-subtext">选中模型将优先展示，便于对照训练结果、可视化图像与权重文件。</div>
              </div>

              <div class="experiment-list">
                <div
                  v-for="experiment in tab.group.experiments"
                  :key="experiment.job_id"
                  class="experiment-row"
                  :class="{ selected: experiment.selected }"
                >
                  <div class="experiment-main">
                    <div class="experiment-topline">
                      <div class="experiment-ident">
                        <span class="tree-dot" :class="{
                          'tree-dot-prod': experiment.current_production,
                          'tree-dot-selected': experiment.selected,
                        }"></span>
                        <div class="experiment-text">
                          <strong>{{ experiment.job_id }}</strong>
                          <span class="experiment-name">{{ experiment.title }}</span>
                        </div>
                      </div>
                      <div class="experiment-tags">
                        <el-tag v-if="experiment.selected" size="small" type="success">当前展示</el-tag>
                        <el-tag size="small" :type="experiment.current_production ? 'success' : statusTag(experiment.status)">
                          {{ experiment.current_production ? '生产中' : experiment.status }}
                        </el-tag>
                      </div>
                    </div>

                    <div class="experiment-metrics">
                      <span
                        v-for="metric in experiment.summary_metrics"
                        :key="`${experiment.job_id}-${metric.label}`"
                        class="metric-chip"
                      >
                        <span class="metric-key">{{ metric.label }}</span>
                        <strong class="metric-val">{{ metric.value }}</strong>
                      </span>
                    </div>
                  </div>

                  <div class="artifact-links">
                    <div v-if="imageArtifacts(experiment).length > 0" class="artifact-previews">
                      <a
                        v-for="artifact in imageArtifacts(experiment)"
                        :key="`${experiment.job_id}-preview-${artifact.url}`"
                        :href="toApiUrl(artifact.url)"
                        target="_blank"
                        rel="noreferrer"
                        class="artifact-preview-card"
                      >
                        <img :src="toApiUrl(artifact.url)" :alt="artifact.name" />
                        <span>{{ artifact.name }}</span>
                      </a>
                    </div>

                    <div v-if="fileArtifacts(experiment).length > 0" class="artifact-file-list">
                      <a
                        v-for="artifact in fileArtifacts(experiment)"
                        :key="`${experiment.job_id}-${artifact.url}`"
                        :href="toApiUrl(artifact.url)"
                        target="_blank"
                        rel="noreferrer"
                      >
                        {{ artifact.name }}
                      </a>
                    </div>
                  </div>
                </div>

                <div v-if="(tab.group.experiments || []).length === 0" class="empty-tip">
                  暂无任何衍生实验记录。
                </div>
              </div>
            </div>
          </div>
        </el-tab-pane>
      </el-tabs>
    </div>
  </section>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import MetricChartPanel from '../charts/MetricChartPanel.vue'
import { toApiUrl } from '../../api/index'

const props = defineProps({
  showcaseData: {
    type: Object,
    default: () => ({ cards: [], experiment_groups: [] }),
  },
  analysisModelOptions: {
    type: Object,
    default: () => ({
      detection: { options: [], current_value: '' },
      recognition: { options: [], current_value: '' },
      tracker: { options: [], current_value: '' },
    }),
  },
  analysisModelSelection: {
    type: Object,
    default: () => ({
      detection_model_path: '',
      recognition_model_path: '',
      tracker_config_path: '',
    }),
  },
})

const emit = defineEmits(['refresh', 'model-selection-change'])

const activeTab = ref('')
const selectedDetectionModel = ref('')
const selectedRecognitionModel = ref('')
const selectedTrackerConfig = ref('')

const showcaseTabs = computed(() => {
  const cards = props.showcaseData.cards || []
  const groups = props.showcaseData.experiment_groups || []
  const groupMap = new Map(groups.map((group) => [group.task_type, group]))

  return cards.map((card) => ({
    key: card.key,
    title: card.title,
    card,
    group: groupMap.get(card.key) || null,
  }))
})

const syncSelectionState = () => {
  selectedDetectionModel.value =
    props.analysisModelSelection?.detection_model_path ||
    props.analysisModelOptions?.detection?.current_value ||
    props.analysisModelOptions?.detection?.options?.[0]?.value ||
    ''
  selectedRecognitionModel.value =
    props.analysisModelSelection?.recognition_model_path ||
    props.analysisModelOptions?.recognition?.current_value ||
    props.analysisModelOptions?.recognition?.options?.[0]?.value ||
    ''
  selectedTrackerConfig.value =
    props.analysisModelSelection?.tracker_config_path ||
    props.analysisModelOptions?.tracker?.current_value ||
    props.analysisModelOptions?.tracker?.options?.[0]?.value ||
    ''
}

const emitModelSelection = () => {
  emit('model-selection-change', {
    detection_model_path:
      selectedDetectionModel.value || props.analysisModelOptions?.detection?.current_value || '',
    recognition_model_path:
      selectedRecognitionModel.value || props.analysisModelOptions?.recognition?.current_value || '',
    tracker_config_path:
      selectedTrackerConfig.value || props.analysisModelOptions?.tracker?.current_value || '',
  })
}

watch(
  () => [props.analysisModelSelection, props.analysisModelOptions],
  () => {
    syncSelectionState()
  },
  { deep: true, immediate: true },
)

watch(
  () => showcaseTabs.value,
  (tabs) => {
    if (!tabs.length) {
      activeTab.value = ''
      return
    }
    if (!tabs.some((tab) => tab.key === activeTab.value)) {
      activeTab.value = tabs[0].key
    }
  },
  { immediate: true },
)

const statusTag = (status) => {
  if (status === 'completed') return 'success'
  if (status === 'running') return 'warning'
  if (status === 'failed' || status === 'cancelled') return 'danger'
  return 'info'
}

const imageArtifacts = (experiment) => (experiment?.artifacts || []).filter((item) => item.kind === 'image').slice(0, 6)
const fileArtifacts = (experiment) => (experiment?.artifacts || []).filter((item) => item.kind !== 'image').slice(0, 8)
</script>

<style scoped>
.showcase-section {
  height: 100%;
  min-height: 0;
}

.selection-panel {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin-bottom: 18px;
}

.selection-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 14px 16px;
  border: 1px solid var(--border-color);
  border-radius: 12px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(248, 250, 252, 0.96));
}

.selection-card span {
  font-size: 13px;
  font-weight: 700;
  color: var(--text-primary);
}

.selection-select {
  width: 100%;
}

.showcase-container {
  margin-top: 10px;
}

:deep(.showcase-tabs .el-tabs__item) {
  font-size: 16px;
  font-weight: 500;
  padding: 0 20px;
  height: 48px;
  line-height: 48px;
}

:deep(.showcase-tabs .el-tabs__item.is-active) {
  font-weight: 700;
}

:deep(.showcase-tabs .el-tabs__nav-wrap::after) {
  height: 1px;
  background-color: var(--border-color);
}

.production-crown {
  padding: 24px 0 32px;
}

.showcase-top,
.experiment-topline {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 14px;
}

.crown-badge {
  font-size: 12px;
  font-weight: 800;
  color: #10a37f;
  letter-spacing: 0.5px;
  margin-bottom: 6px;
}

.showcase-label {
  font-size: 22px;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 8px;
}

.showcase-artifact {
  color: var(--text-secondary);
  font-family: monospace;
  font-size: 14px;
}

.selection-hint {
  margin: 8px 0 0;
  max-width: 720px;
  color: var(--text-secondary);
  line-height: 1.7;
  font-size: 13px;
}

.showcase-side {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 8px;
}

.showcase-key {
  font-size: 12px;
  font-weight: 700;
  color: var(--text-secondary);
  letter-spacing: 0.08em;
}

.production-pill {
  padding: 6px 10px;
  border-radius: 999px;
  background: #ecfdf5;
  color: #047857;
  font-size: 12px;
  font-weight: 700;
}

.showcase-summary {
  margin: 18px 0 0;
  color: var(--text-secondary);
  line-height: 1.8;
  font-size: 14px;
  max-width: 85%;
}

.showcase-block {
  margin-top: 24px;
}

.showcase-block-title {
  margin-bottom: 12px;
  font-size: 14px;
  font-weight: 700;
  color: var(--text-primary);
}

.showcase-list {
  margin: 0;
  padding-left: 18px;
  color: var(--text-secondary);
  line-height: 1.8;
  font-size: 14px;
}

.showcase-stat-grid {
  margin-top: 30px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 16px;
}

.metric-section-grid,
.chart-gallery {
  margin-top: 18px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 16px;
}

.showcase-stat-box {
  padding: 20px;
  border-radius: 12px;
  background: transparent;
  border: 1px solid var(--border-color);
}

.showcase-highlight {
  background: #f8fafc;
}

.showcase-stat-list,
.experiment-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.showcase-stat-item {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  color: var(--text-secondary);
  font-size: 13px;
}

.showcase-stat-item strong {
  color: var(--text-primary);
  text-align: right;
  word-break: break-all;
}

.visual-block {
  margin-top: 22px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.visual-group {
  padding: 18px;
  border: 1px solid var(--border-color);
  border-radius: 14px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(248, 250, 252, 0.92));
}

.visual-head {
  margin-bottom: 12px;
}

.visual-desc {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.6;
}

.visual-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 12px;
}

.visual-card {
  display: flex;
  flex-direction: column;
  gap: 6px;
  text-decoration: none;
  color: var(--text-primary);
}

.visual-card img {
  width: 100%;
  height: 132px;
  object-fit: cover;
  border-radius: 10px;
  border: 1px solid var(--border-color);
  background: var(--surface-secondary);
}

.visual-card strong {
  font-size: 13px;
}

.visual-card span {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.5;
}

.history-block {
  border-top: 1px solid var(--border-color);
  padding: 32px 0 10px;
}

.history-title {
  font-size: 18px;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 6px;
}

.history-subtext {
  font-size: 13px;
  color: var(--text-secondary);
  margin-bottom: 24px;
}

.experiment-row {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 18px;
  padding: 18px 0;
  border-bottom: 1px dashed var(--border-color);
}

.experiment-row.selected {
  background: linear-gradient(90deg, rgba(16, 163, 127, 0.05), transparent 72%);
}

.experiment-row:last-child {
  border-bottom: none;
}

.experiment-main {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
  flex: 1;
}

.experiment-ident {
  display: flex;
  align-items: center;
  gap: 10px;
}

.experiment-text {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.experiment-name {
  font-size: 12px;
  color: var(--text-secondary);
}

.experiment-tags {
  display: flex;
  gap: 8px;
  align-items: center;
}

.tree-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  border: 2px solid #cbd5e1;
}

.tree-dot-prod {
  border-color: #10a37f;
  background-color: #10a37f;
}

.tree-dot-selected {
  border-color: #2563eb;
  background-color: #2563eb;
}

.experiment-ident strong {
  font-size: 15px;
  color: var(--text-primary);
}

.experiment-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  padding-left: 20px;
}

.metric-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: white;
  border: 1px solid #e2e8f0;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 12px;
}

.metric-key {
  color: var(--text-secondary);
}

.metric-val {
  color: var(--text-primary);
  font-family: monospace;
}

.artifact-links {
  display: flex;
  flex-direction: column;
  gap: 10px;
  width: min(48%, 520px);
}

.artifact-previews {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 10px;
  width: 100%;
}

.artifact-preview-card {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 8px;
  border: 1px solid var(--border-color);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.82);
  color: var(--text-primary);
  text-decoration: none;
  text-align: left;
}

.artifact-preview-card img {
  width: 100%;
  height: 96px;
  object-fit: cover;
  border-radius: 8px;
  background: var(--surface-secondary);
}

.artifact-preview-card span {
  font-size: 12px;
  line-height: 1.4;
  word-break: break-word;
}

.artifact-file-list {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 14px;
}

.artifact-file-list a {
  color: var(--accent-primary);
  text-decoration: none;
  font-size: 13px;
  font-weight: 600;
}

.empty-tip {
  color: var(--text-secondary);
  font-size: 14px;
}

@media (max-width: 980px) {
  .selection-panel {
    grid-template-columns: 1fr;
  }

  .experiment-row {
    flex-direction: column;
  }

  .artifact-links {
    width: 100%;
  }

  .showcase-summary {
    max-width: 100%;
  }
}
</style>
