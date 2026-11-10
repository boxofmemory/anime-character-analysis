<template>
  <section class="section-card training-section">
    <div class="section-header">
      <div>
        <div class="eyebrow">Training Hub</div>
        <h2>模型训练</h2>
        <p>统一管理检测、识别与跟踪模型训练任务，支持日志查看、结果评估和生产模型切换。</p>
      </div>
      <el-button type="primary" plain @click="refreshAll">刷新训练状态</el-button>
    </div>

    <el-skeleton :loading="loading" animated :rows="12">
      <template #default>
        <div class="training-container">
          <el-tabs v-model="activeTaskType" class="training-tabs">
            <el-tab-pane
              v-for="task in trainingTasks"
              :key="task.task_type"
              :label="task.title"
              :name="task.task_type"
            >
              <article class="training-card">
                <div class="training-top">
              <div>
                <div class="training-title">{{ task.title }}</div>
                <div class="training-desc">{{ task.description }}</div>
              </div>
              <span class="task-badge">{{ task.task_type.toUpperCase() }}</span>
            </div>

            <div class="production-strip">
              <div>
                <span class="strip-label">当前生产模型</span>
                <strong>{{ task.current_production || '-' }}</strong>
              </div>
              <div class="strip-subtext">
                {{ task.current_production_experiment_id ? `来源实验：${task.current_production_experiment_id}` : '尚未记录晋升来源' }}
              </div>
            </div>

            <div class="form-grid training-form-grid">
              <div class="form-field">
                <label>任务标题</label>
                <el-input v-model="forms[task.task_type].display_name" placeholder="输入本次训练说明" />
              </div>
              <div class="form-field">
                <label>数据集预置</label>
                <el-select
                  :model-value="forms[task.task_type].datasetPreset"
                  placeholder="选择预置路径"
                  @change="(value) => applyDatasetPreset(task.task_type, value)"
                >
                  <el-option
                    v-for="preset in task.dataset_presets"
                    :key="`${task.task_type}-${preset.path}`"
                    :label="`${preset.label}${preset.exists ? '' : '（缺失）'}`"
                    :value="preset.path"
                  />
                </el-select>
              </div>
              <div class="form-field span-2">
                <label>数据集路径</label>
                <el-input v-model="forms[task.task_type].dataset_path" placeholder="可以改成本地其他路径" />
              </div>
              <div v-if="task.primary_resource_field" class="form-field">
                <label>{{ fieldLabel(task.primary_resource_field) }}</label>
                <el-select
                  :model-value="forms[task.task_type].primary_resource_path"
                  placeholder="选择当前资源"
                  filterable
                  allow-create
                  @change="(value) => (forms[task.task_type].primary_resource_path = value)"
                >
                  <el-option
                    v-for="item in task.primary_resource_options"
                    :key="`${task.task_type}-${item.path}`"
                    :label="item.label"
                    :value="item.path"
                  />
                </el-select>
              </div>
              <div v-if="task.secondary_resource_field" class="form-field">
                <label>{{ fieldLabel(task.secondary_resource_field) }}</label>
                <el-select
                  :model-value="forms[task.task_type].secondary_resource_path"
                  placeholder="选择检测模型"
                  filterable
                  allow-create
                  @change="(value) => (forms[task.task_type].secondary_resource_path = value)"
                >
                  <el-option
                    v-for="item in task.secondary_resource_options"
                    :key="`${task.task_type}-${item.path}`"
                    :label="item.label"
                    :value="item.path"
                  />
                </el-select>
              </div>
            </div>

            <div class="core-fields">
              <div class="subsection-title">推荐参数</div>
              <div class="core-grid">
                <template v-for="field in editableCoreFields(task)" :key="`${task.task_type}-${field.key}`">
                  <div class="form-field">
                    <label>{{ field.label }}</label>
                    <el-switch
                      v-if="field.type === 'boolean'"
                      v-model="forms[task.task_type].core[field.key]"
                    />
                    <el-input-number
                      v-else-if="field.type === 'number'"
                      v-model="forms[task.task_type].core[field.key]"
                      :step="numberStep(forms[task.task_type].core[field.key])"
                      controls-position="right"
                    />
                    <el-input
                      v-else
                      v-model="forms[task.task_type].core[field.key]"
                      :placeholder="field.label"
                    />
                  </div>
                </template>
              </div>
            </div>

            <el-collapse class="advanced-collapse">
              <el-collapse-item title="高级参数覆盖（JSON）" name="advanced">
                <el-input
                  v-model="forms[task.task_type].advanced_json"
                  type="textarea"
                  :rows="8"
                  placeholder='例如：{"workers": 2, "patience": 30}'
                />
              </el-collapse-item>
              <el-collapse-item
                v-if="task.task_type === 'tracker'"
                title="ByteTrack YAML 参数覆盖（JSON）"
                name="tracker-yaml"
              >
                <el-input
                  v-model="forms[task.task_type].tracker_yaml_json"
                  type="textarea"
                  :rows="8"
                  placeholder='例如：{"track_buffer": 80, "match_thresh": 0.78}'
                />
              </el-collapse-item>
            </el-collapse>

            <div class="training-actions">
              <el-button
                type="primary"
                :loading="submittingTask === task.task_type"
                @click="submitTask(task)"
              >
                启动{{ task.title }}
              </el-button>
            </div>

            <div class="runtime-block">
              <div class="subsection-title">任务状态与日志</div>
              <div v-if="latestJob(task.task_type)" class="job-runtime-card">
                <div class="job-topline">
                  <div>
                    <strong>{{ latestJob(task.task_type).display_name || latestJob(task.task_type).job_id }}</strong>
                    <span class="job-id">{{ latestJob(task.task_type).job_id }}</span>
                  </div>
                  <el-tag :type="statusTag(latestJob(task.task_type).status)">
                    {{ latestJob(task.task_type).status }}
                  </el-tag>
                </div>
                <div class="job-metrics">
                  <span v-for="metric in latestJob(task.task_type).summary_metrics" :key="metric.label" class="metric-chip">
                    {{ metric.label }}：{{ metric.value }}
                  </span>
                </div>
                <div class="job-actions">
                  <el-button text @click="openDetail(latestJob(task.task_type).job_id)">查看结果</el-button>
                  <el-button
                    v-if="latestJob(task.task_type).status === 'running'"
                    text
                    type="danger"
                    @click="cancelJob(latestJob(task.task_type).job_id)"
                  >
                    取消任务
                  </el-button>
                  <el-button
                    v-if="latestJob(task.task_type).status === 'completed' && !latestJob(task.task_type).current_production"
                    text
                    type="primary"
                    @click="promoteJob(latestJob(task.task_type).job_id)"
                  >
                    设为当前生产模型
                  </el-button>
                </div>
                <pre class="job-log">{{ logCache[latestJob(task.task_type).job_id] || '暂无日志内容' }}</pre>
              </div>
              <div v-else class="empty-state training-empty">还没有这类训练任务，先提交一次任务就会在这里显示。</div>
            </div>

            <div class="result-block">
              <div class="subsection-title">最近实验结果</div>
              <div class="result-list">
                <article
                  v-for="job in recentJobs(task.task_type)"
                  :key="job.job_id"
                  class="result-item"
                >
                  <div class="result-copy">
                    <strong>{{ job.display_name || job.job_id }}</strong>
                    <span>{{ job.updated_at || job.created_at || '-' }}</span>
                  </div>
                  <div class="result-side">
                    <el-tag size="small" :type="job.current_production ? 'success' : statusTag(job.status)">
                      {{ job.current_production ? '生产中' : job.status }}
                    </el-tag>
                    <el-button text @click="openDetail(job.job_id)">查看</el-button>
                  </div>
                </article>
                <div v-if="recentJobs(task.task_type).length === 0" class="empty-tip">暂无实验结果。</div>
              </div>
            </div>
          </article>
            </el-tab-pane>
          </el-tabs>
        </div>
      </template>
    </el-skeleton>

    <el-drawer v-model="detailVisible" size="58%" :destroy-on-close="false">
      <template #header>
        <div class="detail-header">
          <div>
            <h3>{{ selectedJob?.display_name || selectedJob?.job_id || '训练结果' }}</h3>
            <p>{{ selectedJob?.title }} / {{ selectedJob?.status }}</p>
          </div>
        </div>
      </template>

      <div v-if="selectedJobDetail" class="detail-body">
        <div class="detail-metrics">
          <div v-for="metric in selectedJobDetail.summary_metrics" :key="metric.label" class="detail-metric-chip">
            <span>{{ metric.label }}</span>
            <strong>{{ metric.value }}</strong>
          </div>
        </div>

        <div class="artifact-grid">
          <a
            v-for="artifact in imageArtifacts"
            :key="artifact.relative_path"
            class="artifact-image-card"
            :href="toApiUrl(artifact.url)"
            target="_blank"
            rel="noreferrer"
          >
            <img :src="toApiUrl(artifact.url)" :alt="artifact.name" />
            <span>{{ artifact.name }}</span>
          </a>
        </div>

        <div class="artifact-list">
          <article v-for="artifact in fileArtifacts" :key="artifact.relative_path" class="artifact-row">
            <div>
              <strong>{{ artifact.name }}</strong>
              <span>{{ artifact.kind }} / {{ formatBytes(artifact.size_bytes) }}</span>
            </div>
            <a :href="toApiUrl(artifact.url)" target="_blank" rel="noreferrer">下载 / 查看</a>
          </article>
        </div>

        <div v-if="selectedJobDetail.summary_markdown" class="markdown-panel">
          <div class="subsection-title">实验摘要</div>
          <pre>{{ selectedJobDetail.summary_markdown }}</pre>
        </div>

        <div class="markdown-panel">
          <div class="subsection-title">任务日志</div>
          <pre>{{ selectedJobLog }}</pre>
        </div>
      </div>
    </el-drawer>
  </section>
</template>

<script setup>
import { computed, onMounted, onUnmounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import {
  apiCancelTrainingJob,
  apiCreateTrainingJob,
  apiGetTrainingJob,
  apiGetTrainingJobs,
  apiGetTrainingLog,
  apiGetTrainingOptions,
  apiPromoteTrainingJob,
  toApiUrl,
} from '../../api/index'

const emit = defineEmits(['showcase-refresh'])

const loading = ref(true)
const activeTaskType = ref('')
const submittingTask = ref('')
const options = ref({ tasks: [] })
const jobs = ref([])
const forms = reactive({})
const logCache = reactive({})
const detailVisible = ref(false)
const selectedJob = ref(null)
const selectedJobDetail = ref(null)
const selectedJobLog = ref('')
let pollTimer = null

const fieldLabelMap = {
  dataset_yaml: 'YOLO 数据集 YAML',
  dataset_dir: 'ArcFace 数据集目录',
  videos_dir: '评估视频目录',
  base_model: '基础 YOLO 权重',
  pretrained_backbone: '预训练特征权重',
  tracker_configs: 'ByteTrack YAML',
  detector_model: '检测模型',
  epochs: '训练轮次',
  batch: 'Batch Size',
  batch_size: 'Batch Size',
  imgsz: '输入尺寸',
  learning_rate: '学习率',
  device: '设备',
  patience: 'Patience',
  embedding_size: 'Embedding',
  input_size: '输入尺寸',
  max_videos: '评估视频数',
  confidence: '置信度阈值',
  iou: 'IOU 阈值',
}

const trainingTasks = computed(() => options.value.tasks || [])

const imageArtifacts = computed(() =>
  (selectedJobDetail.value?.artifacts || []).filter((item) => item.kind === 'image'),
)

const fileArtifacts = computed(() =>
  (selectedJobDetail.value?.artifacts || []).filter((item) => item.kind !== 'image'),
)

const fieldLabel = (key) => fieldLabelMap[key] || key

const numberStep = (value) => {
  if (Number.isInteger(value)) return 1
  return 0.01
}

const latestJob = (taskType) => jobs.value.find((item) => item.task_type === taskType) || null

const recentJobs = (taskType) => jobs.value.filter((item) => item.task_type === taskType).slice(0, 4)

const editableCoreFields = (task) => {
  const skip = new Set([task.dataset_field, task.primary_resource_field, task.secondary_resource_field].filter(Boolean))
  return Object.entries(forms[task.task_type]?.core || {})
    .filter(([key]) => !skip.has(key))
    .map(([key, value]) => ({
      key,
      label: fieldLabel(key),
      type: typeof value === 'boolean' ? 'boolean' : typeof value === 'number' ? 'number' : 'text',
    }))
}

const applyDatasetPreset = (taskType, path) => {
  forms[taskType].datasetPreset = path
  forms[taskType].dataset_path = path
}

const buildForms = () => {
  for (const task of trainingTasks.value) {
    const defaultConfig = { ...(task.default_config || {}) }
    const datasetPath = defaultConfig[task.dataset_field] || task.dataset_presets?.[0]?.path || ''
    const primaryValue = task.primary_resource_field ? defaultConfig[task.primary_resource_field] : ''
    const primary = Array.isArray(primaryValue)
      ? primaryValue[0] || task.primary_resource_options?.[0]?.path || ''
      : primaryValue || task.primary_resource_options?.[0]?.path || ''
    const secondary = task.secondary_resource_field ? defaultConfig[task.secondary_resource_field] || task.secondary_resource_options?.[0]?.path || '' : ''
    const core = {}
    Object.entries(defaultConfig).forEach(([key, value]) => {
      if (['experiment_name', 'experiment_id_override'].includes(key)) return
      if (task.task_type === 'tracker' && key === 'tracker_configs') return
      core[key] = value
    })
    forms[task.task_type] = {
      display_name: task.title,
      datasetPreset: datasetPath,
      dataset_path: datasetPath,
      primary_resource_path: primary,
      secondary_resource_path: secondary,
      core,
      advanced_json: '{}',
      tracker_yaml_json: '{}',
    }
  }
  if (!activeTaskType.value && trainingTasks.value.length > 0) {
    activeTaskType.value = trainingTasks.value[0].task_type
  }
}

const parseJson = (value, label) => {
  if (!value?.trim()) return {}
  try {
    const parsed = JSON.parse(value)
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new Error('must be object')
    }
    return parsed
  } catch (error) {
    throw new Error(`${label} 不是合法的 JSON 对象`)
  }
}

const buildOverrides = (task) => {
  const form = forms[task.task_type]
  const coreOverrides = { ...form.core }
  if (task.dataset_field) delete coreOverrides[task.dataset_field]
  if (task.primary_resource_field) delete coreOverrides[task.primary_resource_field]
  if (task.secondary_resource_field) delete coreOverrides[task.secondary_resource_field]
  const advancedOverrides = parseJson(form.advanced_json, '高级参数覆盖')
  const merged = { ...coreOverrides, ...advancedOverrides }
  if (task.task_type === 'tracker') {
    merged.tracker_yaml_overrides = parseJson(form.tracker_yaml_json, 'ByteTrack YAML 参数覆盖')
  }
  return merged
}

const submitTask = async (task) => {
  try {
    submittingTask.value = task.task_type
    const form = forms[task.task_type]
    const response = await apiCreateTrainingJob({
      task_type: task.task_type,
      display_name: form.display_name,
      dataset_path: form.dataset_path,
      base_model_or_tracker_config: form.primary_resource_path || null,
      secondary_resource_path: form.secondary_resource_path || null,
      config_overrides: buildOverrides(task),
    })
    ElMessage.success(`${task.title} 已加入后台任务`)
    await refreshJobs()
    await refreshLog(response.data.job_id)
  } catch (error) {
    ElMessage.error(error?.response?.data?.detail || error.message || '提交训练任务失败')
  } finally {
    submittingTask.value = ''
  }
}

const refreshOptions = async () => {
  const response = await apiGetTrainingOptions()
  options.value = response.data
  buildForms()
}

const refreshJobs = async () => {
  const response = await apiGetTrainingJobs()
  jobs.value = response.data.jobs || []
}

const refreshLog = async (jobId) => {
  const response = await apiGetTrainingLog(jobId)
  logCache[jobId] = response.data?.text || ''
  if (selectedJob.value?.job_id === jobId) {
    selectedJobLog.value = logCache[jobId]
  }
}

const openDetail = async (jobId) => {
  const response = await apiGetTrainingJob(jobId)
  selectedJob.value = response.data
  selectedJobDetail.value = response.data
  detailVisible.value = true
  await refreshLog(jobId)
}

const promoteJob = async (jobId) => {
  try {
    await apiPromoteTrainingJob(jobId)
    ElMessage.success('已设为当前生产模型')
    await refreshAll()
    emit('showcase-refresh')
  } catch (error) {
    ElMessage.error(error?.response?.data?.detail || '晋升生产模型失败')
  }
}

const cancelJob = async (jobId) => {
  try {
    await apiCancelTrainingJob(jobId)
    ElMessage.success('任务已取消')
    await refreshAll()
  } catch (error) {
    ElMessage.error(error?.response?.data?.detail || '取消任务失败')
  }
}

const startPolling = () => {
  stopPolling()
  pollTimer = setInterval(async () => {
    const hasRunning = jobs.value.some((item) => ['pending', 'running'].includes(item.status))
    if (!hasRunning) {
      stopPolling()
      return
    }
    await refreshJobs()
    const runningIds = jobs.value
      .filter((item) => ['pending', 'running'].includes(item.status))
      .map((item) => item.job_id)
      .slice(0, 3)
    await Promise.all(runningIds.map((jobId) => refreshLog(jobId)))
    if (selectedJob.value?.job_id) {
      const detail = await apiGetTrainingJob(selectedJob.value.job_id)
      selectedJobDetail.value = detail.data
      selectedJob.value = detail.data
    }
  }, 5000)
}

const stopPolling = () => {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

const refreshAll = async () => {
  loading.value = true
  try {
    await Promise.all([refreshOptions(), refreshJobs()])
    const runningIds = jobs.value
      .filter((item) => ['pending', 'running'].includes(item.status))
      .map((item) => item.job_id)
      .slice(0, 3)
    await Promise.all(runningIds.map((jobId) => refreshLog(jobId)))
    if (jobs.value.some((item) => ['pending', 'running'].includes(item.status))) {
      startPolling()
    } else {
      stopPolling()
    }
  } finally {
    loading.value = false
  }
}

const formatBytes = (sizeBytes) => {
  if (sizeBytes == null) return '-'
  if (sizeBytes < 1024) return `${sizeBytes} B`
  if (sizeBytes < 1024 * 1024) return `${(sizeBytes / 1024).toFixed(1)} KB`
  return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`
}

const statusTag = (status) => {
  if (status === 'completed') return 'success'
  if (status === 'running') return 'warning'
  if (status === 'failed' || status === 'cancelled') return 'danger'
  return 'info'
}

onMounted(async () => {
  await refreshAll()
})

onUnmounted(() => {
  stopPolling()
})
</script>

<style scoped>
.training-section {
  height: 100%;
  min-height: 0;
}

.training-container {
  margin-top: 10px;
}

.form-field {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 8px;
}

.form-field label {
  color: var(--text-secondary);
  font-size: 13px;
  margin: 0;
}

.form-field .el-input-number,
.form-field .el-select,
.form-field .el-input {
  width: 100%;
}

:deep(.training-tabs .el-tabs__item) {
  font-size: 16px;
  font-weight: 500;
  padding: 0 20px;
  height: 48px;
  line-height: 48px;
}

:deep(.training-tabs .el-tabs__item.is-active) {
  font-weight: 700;
}

:deep(.training-tabs .el-tabs__nav-wrap::after) {
  height: 1px;
  background-color: var(--border-color);
}

.training-card {
  padding: 10px 0;
  display: flex;
  flex-direction: column;
  gap: 22px;
}

.training-top,
.job-topline,
.artifact-row,
.result-item,
.training-actions {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
}

.training-title {
  font-size: 20px;
  font-weight: 700;
  color: var(--text-primary);
}

.training-desc,
.strip-subtext,
.job-id,
.result-copy span,
.artifact-row span {
  color: var(--text-secondary);
  font-size: 13px;
}

.task-badge {
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(75, 120, 255, 0.1);
  color: var(--accent-primary);
  font-size: 12px;
  font-weight: 700;
}

.production-strip {
  padding: 14px 18px;
  border-radius: 8px;
  background: rgba(16, 163, 127, 0.05);
  border: 1px solid rgba(16, 163, 127, 0.15);
  color: var(--text-primary);
}

.job-runtime-card,
.result-block,
.core-fields,
.markdown-panel {
  background: transparent;
  padding-top: 10px;
}

.result-block,
.core-fields,
.runtime-block {
  padding-top: 24px;
  border-top: 1px solid var(--border-color);
}

.strip-label,
.subsection-title {
  display: block;
  margin-bottom: 8px;
  font-size: 13px;
  color: var(--text-secondary);
  font-weight: 700;
}

.training-form-grid {
  margin-bottom: 0;
  gap: 24px 20px;
}

.span-2 {
  grid-column: span 2;
}

.core-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 26px 20px;
}

.advanced-collapse {
  margin-top: -4px;
}

.metric-chip {
  display: inline-flex;
  padding: 5px 10px;
  border-radius: 999px;
  background: rgba(16, 163, 127, 0.08);
  color: var(--text-primary);
  font-size: 12px;
  margin-right: 8px;
  margin-bottom: 8px;
}

.job-actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.job-log,
.markdown-panel pre {
  margin: 12px 0 0;
  border-radius: 10px;
  background: #111827;
  color: #e5eefb;
  padding: 14px;
  font-size: 12px;
  line-height: 1.6;
  max-height: 220px;
  overflow: auto;
  white-space: pre-wrap;
}

.result-list,
.artifact-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.result-item {
  padding: 12px 14px;
  border-radius: 10px;
  border: 1px solid var(--border-color);
  background: #ffffff;
}

.result-copy {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.result-side {
  display: flex;
  align-items: center;
  gap: 10px;
}

.detail-body {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.detail-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.detail-metric-chip {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 160px;
  padding: 12px 14px;
  border-radius: 12px;
  background: #f8fafc;
  border: 1px solid var(--border-color);
}

.artifact-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 12px;
}

.artifact-image-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
  text-decoration: none;
  color: var(--text-primary);
  padding: 10px;
  border-radius: 12px;
  border: 1px solid var(--border-color);
  background: #ffffff;
}

.artifact-image-card img {
  width: 100%;
  aspect-ratio: 1.2 / 1;
  object-fit: cover;
  border-radius: 10px;
  background: #eef2f7;
}

.artifact-row {
  padding: 12px 14px;
  border-radius: 12px;
  border: 1px solid var(--border-color);
  background: #ffffff;
}

.artifact-row a {
  color: var(--accent-primary);
  text-decoration: none;
  font-weight: 600;
}

.training-empty {
  margin-top: 0;
}

@media (max-width: 800px) {
  .core-grid,
  .training-form-grid {
    grid-template-columns: 1fr;
  }

  .span-2 {
    grid-column: auto;
  }
}
</style>
