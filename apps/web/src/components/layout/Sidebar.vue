<template>
  <aside class="sidebar">
    <div class="sidebar-top">
      <div class="brand-card">
        <div class="brand-badge">A</div>
        <div>
          <div class="brand-title">动漫角色分析系统</div>
          <div class="brand-subtitle">视频时序分析与角色库管理</div>
        </div>
      </div>

      <button class="action-item primary" type="button" @click="emit('new-task')">
        <el-icon><Edit /></el-icon>
        <span>新建分析任务</span>
      </button>

      <button class="action-item ghost" type="button" @click="emit('navigate', 'dashboard')">
        <el-icon><DataBoard /></el-icon>
        <span>系统仪表盘</span>
      </button>

      <el-popover
        placement="right-start"
        trigger="click"
        :width="260"
        :fallback-placements="['bottom-start', 'top-start']"
      >
        <template #reference>
          <button class="config-toggle" type="button">
            <el-icon><Operation /></el-icon>
            <div class="config-toggle-content">
              <div class="config-toggle-title">当前分析配置</div>
              <div class="config-toggle-val text-ellipsis" :title="currentConfigSummary">
                {{ currentConfigSummary }}
              </div>
            </div>
            <el-icon class="arrow-icon"><ArrowRight /></el-icon>
          </button>
        </template>
        
        <div class="popover-config-container">
          <div class="popover-title">模型选择与配置</div>

          <div class="selector-row">
            <span class="selector-label">检测模型</span>
            <el-select v-model="currentDetectionModel" size="small" class="compact-select" @change="emitModelSelection">
              <el-option
                v-for="item in modelOptions.detection.options"
                :key="item.value"
                :label="item.label"
                :value="item.value"
              />
            </el-select>
          </div>

          <div class="selector-row">
            <span class="selector-label">特征模型</span>
            <el-select v-model="currentRecognitionModel" size="small" class="compact-select" @change="emitModelSelection">
              <el-option
                v-for="item in modelOptions.recognition.options"
                :key="item.value"
                :label="item.label"
                :value="item.value"
              />
            </el-select>
          </div>

          <div class="selector-row">
            <span class="selector-label">跟踪配置</span>
            <el-select v-model="currentTrackerConfig" size="small" class="compact-select" @change="emitModelSelection">
              <el-option
                v-for="item in modelOptions.tracker.options"
                :key="item.value"
                :label="item.label"
                :value="item.value"
              />
            </el-select>
          </div>

          <div class="selector-hint">
            切换后，新上传的视频会按这套模型组合执行分析。
          </div>
        </div>
      </el-popover>
    </div>

    <div
      class="sidebar-history"
      :class="{ 'is-scrolling': isScrolling }"
      @scroll="handleScroll"
    >
      <div class="history-header">
        <span>你的分析历史</span>
        <el-button text type="primary" @click="loadHistoryList">刷新</el-button>
      </div>

      <div v-if="historyError" class="history-alert">
        {{ historyError }}
      </div>

      <div v-if="historyLoading" class="empty-history">
        正在读取历史分析记录...
      </div>

      <div v-else-if="historyList.length === 0" class="empty-history">
        还没有历史分析记录，先上传一个视频试试看。
      </div>

      <div
        v-for="item in historyList"
        :key="item.id"
        class="history-item-wrapper"
      >
        <button
          type="button"
          class="history-item"
          :class="{ unavailable: !item.has_annotated_video }"
          :title="item.filename"
          @click="handleSelectHistory(item)"
        >
          <el-icon class="history-icon"><Film /></el-icon>
          <div class="history-item-body">
            <span class="text-ellipsis">{{ item.filename }}</span>
            <div class="history-meta">
              <span>{{ formatDuration(item.duration) }}</span>
              <span
                class="history-status"
                :class="item.has_annotated_video ? 'ready' : 'missing'"
              >
                {{ item.has_annotated_video ? '可播放' : '标注缺失' }}
              </span>
            </div>
          </div>
        </button>

        <el-dropdown trigger="click" placement="bottom-end">
          <span class="more-btn"><el-icon><MoreFilled /></el-icon></span>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item @click="handleRename(item)">
                <el-icon><EditPen /></el-icon>重命名
              </el-dropdown-item>
              <el-dropdown-item divided class="danger-item" @click="handleDelete(item.id)">
                <el-icon><Delete /></el-icon>删除分析
              </el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </div>
    </div>

    <div class="sidebar-bottom">
      <el-dropdown trigger="click" placement="top-start" style="width: 100%;">
        <div class="user-item" :class="{ active: currentPage !== 'video' }">
          <div class="avatar"><el-icon><UserFilled /></el-icon></div>
          <div class="user-info">
            <span class="username">系统后台数据</span>
            <span class="plan">设置与仪表盘</span>
          </div>
        </div>
        <template #dropdown>
          <el-dropdown-menu class="bottom-menu">
            <el-dropdown-item @click="emit('navigate', 'dashboard')">
              <el-icon><DataBoard /></el-icon>仪表盘
            </el-dropdown-item>
            <el-dropdown-item @click="emit('navigate', 'settings')">
              <el-icon><Setting /></el-icon>系统设置
            </el-dropdown-item>
            <el-dropdown-item divided @click="emit('navigate', 'video')">
              <el-icon><VideoPlay /></el-icon>返回分析台
            </el-dropdown-item>
            <el-dropdown-item @click="handleLogout">
              <el-icon><SwitchButton /></el-icon>退出登录
            </el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
    </div>
  </aside>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import {
  ArrowRight,
  DataBoard,
  Delete,
  Edit,
  EditPen,
  Film,
  MoreFilled,
  Operation,
  Setting,
  SwitchButton,
  UserFilled,
  VideoPlay,
} from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  apiDeleteVideo,
  apiGetAnalysisModelOptions,
  apiGetHealthReady,
  apiGetHistory,
  apiRenameVideo,
} from '../../api/index'

const props = defineProps({
  currentPage: {
    type: String,
    default: 'video',
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

const emit = defineEmits(['navigate', 'new-task', 'select-history', 'history-deleted', 'model-selection-change'])

const currentDetectionModel = ref('')
const currentRecognitionModel = ref('')
const currentTrackerConfig = ref('')
const historyList = ref([])
const historyLoading = ref(false)
const historyError = ref('')
const modelOptions = reactive({
  detection: { options: [], current_value: '' },
  recognition: { options: [], current_value: '' },
  tracker: { options: [], current_value: '' },
})

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

const isStartupConnectionError = (error) => {
  const status = error?.response?.status
  return !error?.response || status === 502 || status === 503 || status === 504
}

const requestWithStartupRetry = async (requestFn, attempts = 6) => {
  let lastError = null
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      if (attempt > 0) {
        await apiGetHealthReady()
      }
      return await requestFn()
    } catch (error) {
      lastError = error
      if (!isStartupConnectionError(error) || attempt === attempts - 1) {
        break
      }
      await wait(800 + attempt * 500)
    }
  }
  throw lastError
}

const currentConfigSummary = computed(() => {
  const getModelName = (options, val) => {
    if (!options || !val) return ''
    const opt = options.find(o => o.value === val)
    if (!opt) return ''
    const parts = opt.label.split('·')
    return parts.length > 1 ? parts[1].trim() : opt.label
  }
  const det = getModelName(modelOptions.detection.options, currentDetectionModel.value) || '检测'
  const rec = getModelName(modelOptions.recognition.options, currentRecognitionModel.value) || '特征'
  return `${det} + ${rec}`
})

const isScrolling = ref(false)
let scrollTimer = null

const handleScroll = () => {
  isScrolling.value = true
  if (scrollTimer) clearTimeout(scrollTimer)
  scrollTimer = setTimeout(() => {
    isScrolling.value = false
  }, 1500)
}

const applyIncomingSelection = (selection) => {
  if (selection?.detection_model_path) currentDetectionModel.value = selection.detection_model_path
  if (selection?.recognition_model_path) currentRecognitionModel.value = selection.recognition_model_path
  if (selection?.tracker_config_path) currentTrackerConfig.value = selection.tracker_config_path
}

const emitModelSelection = () => {
  emit('model-selection-change', {
    detection_model_path: currentDetectionModel.value || modelOptions.detection.current_value || '',
    recognition_model_path: currentRecognitionModel.value || modelOptions.recognition.current_value || '',
    tracker_config_path: currentTrackerConfig.value || modelOptions.tracker.current_value || '',
  })
}

const loadAnalysisModelOptions = async () => {
  try {
    const response = await requestWithStartupRetry(apiGetAnalysisModelOptions)
    const payload = response.data || {}
    modelOptions.detection = payload.detection || { options: [], current_value: '' }
    modelOptions.recognition = payload.recognition || { options: [], current_value: '' }
    modelOptions.tracker = payload.tracker || { options: [], current_value: '' }

    currentDetectionModel.value =
      props.analysisModelSelection?.detection_model_path ||
      modelOptions.detection.current_value ||
      modelOptions.detection.options?.[0]?.value ||
      ''
    currentRecognitionModel.value =
      props.analysisModelSelection?.recognition_model_path ||
      modelOptions.recognition.current_value ||
      modelOptions.recognition.options?.[0]?.value ||
      ''
    currentTrackerConfig.value =
      props.analysisModelSelection?.tracker_config_path ||
      modelOptions.tracker.current_value ||
      modelOptions.tracker.options?.[0]?.value ||
      ''

    emitModelSelection()
  } catch (error) {
    console.error('无法获取可选分析模型', error)
  }
}

const loadHistoryList = async () => {
  historyLoading.value = true
  historyError.value = '后端服务启动中，正在自动重试...'
  try {
    const response = await requestWithStartupRetry(apiGetHistory)
    historyList.value = response.data
    historyError.value = ''
  } catch (error) {
    historyError.value = '历史记录读取失败，请确认后端服务正在运行。'
    console.error('无法获取历史记录', error)
  } finally {
    historyLoading.value = false
  }
}

const formatDuration = (duration) => {
  if (duration == null || Number.isNaN(Number(duration))) {
    return '时长未知'
  }

  const totalSeconds = Math.max(Math.round(Number(duration)), 0)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}

const handleSelectHistory = (item) => {
  if (!item.has_annotated_video) {
    ElMessage.warning('该历史记录缺少标注视频文件，当前只能查看时间轴数据。')
  }
  emit('navigate', 'video')
  emit('select-history', item)
}

const handleRename = async (item) => {
  try {
    const { value: newName } = await ElMessageBox.prompt(
      '请输入新的文件名，需要保留视频扩展名。',
      '重命名分析记录',
      {
        confirmButtonText: '保存',
        cancelButtonText: '取消',
        inputValue: item.filename,
        inputPattern: /^.+(\.mp4|\.avi|\.webm|\.mkv)$/i,
        inputErrorMessage: '文件名需要包含有效的视频扩展名，例如 .mp4',
      },
    )
    if (newName && newName !== item.filename) {
      await apiRenameVideo(item.id, newName)
      ElMessage.success('重命名成功')
      await loadHistoryList()
    }
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('重命名失败')
    }
  }
}

const handleDelete = async (id) => {
  try {
    await ElMessageBox.confirm(
      '这会删除视频分析记录、标注视频和时间轴数据，确定继续吗？',
      '删除确认',
      {
        confirmButtonText: '确定删除',
        cancelButtonText: '取消',
        type: 'warning',
      },
    )
    await apiDeleteVideo(id)
    ElMessage.success('删除成功')
    await loadHistoryList()
    emit('history-deleted', id)
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败')
    }
  }
}

const handleLogout = () => {
  ElMessage.info('当前为系统演示入口，账号管理模块暂未启用。')
}

onMounted(() => {
  loadAnalysisModelOptions()
  loadHistoryList()
})

watch(
  () => props.analysisModelSelection,
  (selection) => {
    applyIncomingSelection(selection)
  },
  { deep: true },
)
</script>

<style scoped>
.sidebar {
  width: 280px;
  height: 100vh;
  display: flex;
  flex-direction: column;
  padding: 16px 16px 10px 16px;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border-color);
}

.sidebar-top {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.brand-card {
  display: flex;
  gap: 12px;
  padding: 16px 12px;
  border-radius: 8px;
  background: transparent;
}

.brand-badge {
  width: 38px;
  height: 38px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--accent-primary);
  color: #fff;
  font-weight: 700;
  font-size: 18px;
}

.brand-title {
  font-size: 16px;
  font-weight: 700;
  color: var(--text-primary);
}

.brand-subtitle {
  margin-top: 4px;
  font-size: 12px;
  color: var(--text-secondary);
}

.action-item {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 10px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  padding: 12px 14px;
  font-size: 14px;
  font-weight: 500;
  transition: background 0.2s ease, color 0.2s ease;
}

.action-item.primary {
  background: var(--accent-primary);
  color: #fff;
}

.action-item.ghost {
  background: transparent;
  color: var(--text-primary);
  border: 1px solid var(--border-color);
}

.action-item:hover {
  background: #f4f5f9;
}

.action-item.primary:hover {
  opacity: 0.9;
}

.config-toggle {
  display: flex;
  align-items: center;
  gap: 10px;
  background: transparent;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  cursor: pointer;
  padding: 10px 12px;
  transition: background 0.2s ease, border-color 0.2s ease;
  width: 100%;
  text-align: left;
}

.config-toggle:hover {
  background: #f4f5f9;
}

.config-toggle > .el-icon:first-child {
  font-size: 16px;
  color: var(--text-secondary);
}

.config-toggle-content {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.config-toggle-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
}

.config-toggle-val {
  font-size: 11px;
  color: var(--text-secondary);
}

.arrow-icon {
  font-size: 14px;
  color: var(--text-secondary);
  opacity: 0.7;
}

.popover-config-container {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.popover-title {
  font-size: 13px;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 8px;
}

.selector-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.selector-row + .selector-row {
  margin-top: 10px;
}

.selector-label {
  font-size: 12px;
  color: var(--text-secondary);
}

.selector-hint {
  margin-top: 10px;
  font-size: 11px;
  line-height: 1.6;
  color: var(--text-secondary);
}

.compact-select {
  width: 155px;
}

.sidebar-history {
  flex: 1;
  margin-top: 18px;
  padding: 10px;
  border-radius: 8px;
  background: transparent;
  overflow-y: auto;
}

.sidebar-history::-webkit-scrollbar {
  width: 6px;
  background: transparent;
}

.sidebar-history::-webkit-scrollbar-thumb {
  border-radius: 999px;
  background: transparent;
  transition: background 0.3s ease;
}

.sidebar-history:hover::-webkit-scrollbar-thumb,
.sidebar-history.is-scrolling::-webkit-scrollbar-thumb {
  background: rgba(114, 138, 188, 0.28);
}

.history-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
  font-size: 13px;
  font-weight: 600;
}

.empty-history {
  padding: 14px 4px;
  color: var(--text-secondary);
  font-size: 13px;
  line-height: 1.7;
}

.history-alert {
  margin-bottom: 10px;
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(227, 107, 107, 0.10);
  color: var(--accent-danger);
  font-size: 12px;
  line-height: 1.6;
}

.history-item-wrapper {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 0;
}

.history-item {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 10px;
  border: none;
  background: transparent;
  border-radius: 8px;
  padding: 10px 12px;
  text-align: left;
  cursor: pointer;
  color: var(--text-primary);
  transition: background 0.15s ease;
}

.history-item:hover {
  background: #f0f0f0;
}

.history-item.unavailable {
  background: rgba(255, 245, 245, 0.9);
}

.history-icon {
  font-size: 16px;
  color: var(--text-secondary);
}

.history-item-body {
  min-width: 0;
  display: flex;
  flex: 1;
  flex-direction: column;
  gap: 4px;
}

.history-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--text-secondary);
}

.history-status {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
}

.history-status.ready {
  background: rgba(39, 182, 155, 0.12);
  color: var(--accent-secondary);
}

.history-status.missing {
  background: rgba(227, 107, 107, 0.12);
  color: var(--accent-danger);
}

.text-ellipsis {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.more-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: 10px;
  cursor: pointer;
  color: var(--text-secondary);
  opacity: 0;
  transition: opacity 0.2s ease, background 0.2s ease;
}

.history-item-wrapper:hover .more-btn,
.more-btn:active {
  opacity: 1;
}

.more-btn:hover {
  background: rgba(114, 138, 188, 0.15);
}

.sidebar-bottom {
  padding-top: 8px;
  margin-top: 0;
  border-top: 1px solid var(--border-color);
}

.user-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  border-radius: 8px;
  cursor: pointer;
  background: transparent;
  transition: background 0.1s ease;
}

.user-item:hover,
.user-item.active {
  background: rgba(0, 0, 0, 0.05);
}

.user-item:active {
  background: rgba(0, 0, 0, 0.08);
}

.avatar {
  width: 32px;
  height: 32px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f1f5f9;
  color: #718198;
  font-size: 16px;
}

.user-info {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.username {
  font-size: 13px;
  font-weight: 700;
}

.plan {
  font-size: 11px;
  color: var(--text-secondary);
}

:deep(.bottom-menu) {
  min-width: 220px;
  border-radius: 16px;
  padding: 8px 0;
}

:deep(.danger-item) {
  color: var(--accent-danger) !important;
}
</style>
