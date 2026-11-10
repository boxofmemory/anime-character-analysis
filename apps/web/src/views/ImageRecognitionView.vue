<template>
  <main class="main-content">
    <div class="content-area">
      <div
        v-if="status === 'idle'"
        class="welcome-box"
        @click="$refs.imageInput.click()"
        @drop.prevent="handleDrop"
        @dragover.prevent
      >
        <el-icon :size="60" color="#a1a5b7"><PictureFilled /></el-icon>
        <h2>单图角色识别</h2>
        <p>上传单张图片即可完成角色检测与身份匹配。<br />适合快速验证模型、角色库和阈值配置。</p>
        <input
          ref="imageInput"
          type="file"
          accept="image/png,image/jpeg,image/webp"
          style="display: none;"
          @change="handleFileChange"
        />
        <el-button type="primary" size="large" class="upload-btn" @click.stop="$refs.imageInput.click()">选择图片</el-button>

        <div class="config-hint mt-6">
          <span>
            将使用侧边栏当前配置：{{ displayModelName(analysisModelSelection?.detection_model_path) }} +
            {{ displayModelName(analysisModelSelection?.recognition_model_path) }}
          </span>
        </div>
      </div>

      <div v-if="status === 'processing'" class="progress-box">
        <h2>正在调用模型进行识别...</h2>
        <el-skeleton animated :rows="6" />
      </div>

      <div v-if="status === 'completed'" class="result-workspace">
        <div>
          <el-button @click="resetToIdle">
            <el-icon><Back /></el-icon>
            返回重新上传
          </el-button>
        </div>

        <div class="stats-panel">
          <el-row :gutter="20">
            <el-col :span="8">
              <el-card class="stat-card">
                <div class="stat-item">
                  <div class="stat-label">检测到目标</div>
                  <div class="stat-value">{{ detections.length }}</div>
                </div>
              </el-card>
            </el-col>
            <el-col :span="8">
              <el-card class="stat-card">
                <div class="stat-item">
                  <div class="stat-label">已知角色</div>
                  <div class="stat-value known">{{ knownCount }}</div>
                </div>
              </el-card>
            </el-col>
            <el-col :span="8">
              <el-card class="stat-card">
                <div class="stat-item">
                  <div class="stat-label">陌生或未匹配</div>
                  <div class="stat-value unknown">{{ unknownCount }}</div>
                </div>
              </el-card>
            </el-col>
          </el-row>
        </div>

        <section class="diagnostic-panel">
          <div class="diagnostic-item">
            <span>识别模型</span>
            <strong>{{ modelSelection.recognition_model || displayModelName(analysisModelSelection?.recognition_model_path) }}</strong>
          </div>
          <div class="diagnostic-item">
            <span>角色库 Profile</span>
            <strong>{{ diagnostics.recognition_library_profile || modelSelection.recognition_library_profile || '-' }}</strong>
          </div>
          <div class="diagnostic-item">
            <span>库内已知角色</span>
            <strong>{{ diagnosticKnownCount }}</strong>
          </div>
          <div class="diagnostic-item">
            <span>匹配阈值</span>
            <strong>{{ diagnostics.known_threshold != null ? Number(diagnostics.known_threshold).toFixed(2) : '-' }}</strong>
          </div>
        </section>

        <div class="charts-container">
          <div class="chart-row">
            <div class="chart-box image-box">
              <div class="chart-header">
                <h3 class="chart-title">标注结果</h3>
              </div>
              <img :src="annotatedImageUrl" alt="annotated result" class="annotated-image" />
            </div>

            <div class="chart-box result-box">
              <div class="chart-header">
                <h3 class="chart-title">识别列表</h3>
              </div>
              <div class="detection-list">
                <article v-for="(item, index) in detections" :key="`${item.name}-${index}`" class="detection-row">
                  <div class="detection-info">
                    <strong>{{ item.name }}</strong>
                    <div class="detection-meta">
                      <span v-if="item.similarity != null">相似度 {{ item.similarity.toFixed(2) }}</span>
                      <span v-if="item.confidence != null">置信度 {{ item.confidence.toFixed(2) }}</span>
                    </div>
                    <p v-if="item.reason" class="match-reason">{{ item.reason }}</p>
                    <div v-if="item.candidates?.length" class="candidate-list">
                      <span
                        v-for="candidate in item.candidates.slice(0, 3)"
                        :key="candidate.name"
                        class="candidate-chip"
                      >
                        {{ candidate.name }} {{ Number(candidate.similarity).toFixed(2) }}
                      </span>
                    </div>
                  </div>
                  <el-tag :type="item.type === 'known' ? 'success' : item.type === 'stranger' ? 'warning' : 'info'">
                    {{ item.type === 'known' ? '已知角色' : item.type === 'stranger' ? '陌生角色' : '未匹配' }}
                  </el-tag>
                </article>
                <div v-if="detections.length === 0" class="empty-state">未检测到任何目标</div>
                <div v-else-if="topCandidates.length === 0" class="empty-state compact">
                  当前检测目标没有可用候选，请检查角色库是否已按当前模型构建。
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </main>
</template>

<script setup>
import { computed, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { Back, PictureFilled } from '@element-plus/icons-vue'
import { apiRecognizeImage } from '../api/index'

const props = defineProps({
  analysisModelSelection: {
    type: Object,
    default: () => ({
      detection_model_path: '',
      recognition_model_path: '',
      tracker_config_path: '',
    }),
  },
})

const status = ref('idle')
const annotatedImageUrl = ref('')
const detections = ref([])
const modelSelection = ref({})
const diagnostics = ref({})

const knownCount = computed(() => detections.value.filter((item) => item.type === 'known').length)
const unknownCount = computed(() => detections.value.filter((item) => item.type !== 'known').length)
const diagnosticKnownCount = computed(() => diagnostics.value?.known_count ?? 0)
const topCandidates = computed(() => detections.value[0]?.candidates || [])

const displayModelName = (path) => {
  if (!path) {
    return '当前默认模型'
  }
  const parts = String(path).split(/[\\/]/)
  return parts[parts.length - 1] || path
}

const resetToIdle = () => {
  status.value = 'idle'
  annotatedImageUrl.value = ''
  detections.value = []
  modelSelection.value = {}
  diagnostics.value = {}
}

const runRecognition = async (file) => {
  status.value = 'processing'
  detections.value = []
  annotatedImageUrl.value = ''
  modelSelection.value = {}
  diagnostics.value = {}

  try {
    const response = await apiRecognizeImage(file, props.analysisModelSelection)
    const payload = response.data || {}
    detections.value = payload.detections || []
    modelSelection.value = payload.model_selection || {}
    diagnostics.value = payload.diagnostics || {}
    annotatedImageUrl.value = payload.annotated_image_base64
      ? `data:image/png;base64,${payload.annotated_image_base64}`
      : ''
    status.value = 'completed'
    ElMessage.success('单图识别完成')
  } catch (error) {
    status.value = 'idle'
    ElMessage.error(error?.response?.data?.detail || '单图识别失败')
  }
}

const handleFileChange = (event) => {
  const file = event.target.files?.[0]
  if (file) {
    runRecognition(file)
  }
  event.target.value = ''
}

const handleDrop = (event) => {
  const file = event.dataTransfer?.files?.[0]
  if (file) {
    runRecognition(file)
  }
}
</script>

<style scoped>
.main-content { flex: 1; min-height: 0; display: flex; flex-direction: column; overflow: hidden; }
.content-area { flex: 1; min-height: 0; padding: 20px 40px; overflow-y: auto; display: flex; flex-direction: column; align-items: center; position: relative; }

.welcome-box { text-align: center; background: #ffffff; padding: 60px; border-radius: 12px; box-shadow: var(--shadow-light); border: 1px solid var(--border-color); width: 100%; max-width: 600px; margin-top: 10vh; transition: all 0.3s; cursor: pointer; }
.welcome-box:hover { border-color: var(--accent-primary); background-color: #f8fafc; }
.welcome-box h2 { color: var(--text-primary); margin-top: 20px; font-weight: 600; pointer-events: none; }
.welcome-box p { color: var(--text-secondary); margin-bottom: 30px; line-height: 1.6; pointer-events: none; }
.upload-btn { width: 200px; border-radius: 8px; font-weight: bold; }
.config-hint span { color: var(--text-secondary); font-size: 13px; }

.mt-6 { margin-top: 24px; }

.progress-box { width: 100%; max-width: 600px; background: #ffffff; padding: 40px; border-radius: 12px; box-shadow: var(--shadow-light); border: 1px solid var(--border-color); text-align: center; margin-top: 10vh; }
.progress-box h2 { color: var(--text-primary); margin-bottom: 30px; font-weight: 600; }

.result-workspace { width: 100%; max-width: 1400px; display: flex; flex-direction: column; gap: 20px; padding: 10px 0; }
.stats-panel { width: 100%; }

.stat-card { background-color: #ffffff; border-radius: 12px; padding: 20px; box-shadow: var(--shadow-light); border: 1px solid var(--border-color); transition: all 0.2s ease; cursor: default; }
.stat-card:hover { box-shadow: var(--shadow-medium); transform: translateY(-2px); }
.stat-item { text-align: center; }
.stat-label { font-size: 14px; color: var(--text-secondary); margin-bottom: 8px; }
.stat-value { font-size: 24px; font-weight: 600; color: var(--text-primary); }
.stat-value.known { color: var(--accent-secondary); }
.stat-value.unknown { color: #f59e0b; }

.diagnostic-panel { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; padding: 14px; background: #ffffff; border: 1px solid var(--border-color); border-radius: 8px; box-shadow: var(--shadow-light); }
.diagnostic-item { min-width: 0; display: flex; flex-direction: column; gap: 6px; }
.diagnostic-item span { color: var(--text-secondary); font-size: 12px; }
.diagnostic-item strong { color: var(--text-primary); font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

.charts-container { width: 100%; display: flex; flex-direction: column; gap: 24px; }
.chart-row { display: flex; gap: 24px; width: 100%; align-items: stretch; }
.chart-box { background: #ffffff; border-radius: 12px; padding: 24px; box-shadow: var(--shadow-light); border: 1px solid var(--border-color); display: flex; flex-direction: column; }
.image-box { flex: 1.5; }
.result-box { flex: 1; max-height: 700px; overflow-y: auto; }
.chart-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
.chart-title { margin: 0; color: var(--text-primary); font-size: 16px; display: flex; align-items: center; gap: 8px; }

.annotated-image { width: 100%; border-radius: 8px; background: #f4f5f9; object-fit: contain; max-height: 600px; }

.detection-list { display: flex; flex-direction: column; gap: 10px; }
.detection-row { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; padding: 14px; background: #f8fafc; border: 1px solid var(--border-color); border-radius: 8px; }
.detection-info { min-width: 0; }
.detection-info strong { color: var(--text-primary); font-size: 15px; }
.detection-meta { display: flex; gap: 12px; margin-top: 6px; font-size: 12px; color: var(--text-secondary); }
.match-reason { margin: 8px 0 0 0; color: #b45309; font-size: 12px; line-height: 1.5; }
.candidate-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.candidate-chip { max-width: 100%; padding: 3px 8px; border-radius: 999px; background: #eef2ff; color: #3153a4; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.empty-state { padding: 40px; text-align: center; color: var(--text-secondary); }
.empty-state.compact { padding: 16px; font-size: 13px; }

@media (max-width: 1000px) {
  .chart-row { flex-direction: column; }
  .diagnostic-panel { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

@media (max-width: 640px) {
  .content-area { padding: 16px; }
  .welcome-box { padding: 36px 20px; }
  .diagnostic-panel { grid-template-columns: 1fr; }
}
</style>
