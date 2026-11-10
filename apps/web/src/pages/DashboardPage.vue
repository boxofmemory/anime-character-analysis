<template>
  <section class="dashboard-page">
    <div class="dashboard-hero">
      <div class="hero-copy">
        <span class="hero-kicker">System Overview</span>
        <h1>系统仪表盘</h1>
        <p>集中呈现角色库、模型配置、视频分析结果与系统运行概况。</p>
      </div>

      <div class="hero-actions">
        <div class="hero-chip" v-for="chip in heroChips" :key="chip.label">
          <span>{{ chip.label }}</span>
          <strong>{{ chip.value }}</strong>
        </div>
        <el-button class="refresh-btn" type="primary" plain @click="loadSummary">刷新数据</el-button>
      </div>
    </div>

    <el-skeleton :loading="loading" animated :rows="10">
      <template #default>
        <div class="stats-grid">
          <article
            v-for="(stat, index) in summary.stats"
            :key="stat.label"
            class="metric-card interactive"
            @click="handleStatClick(index)"
          >
            <div class="metric-label">{{ stat.label }}</div>
            <div class="metric-value">{{ stat.value }}</div>
            <div class="metric-hint">{{ stat.hint }}</div>
          </article>
        </div>

        <div class="dashboard-grid">
          <div class="main-column">
            <section class="panel trend-panel">
              <div class="panel-head">
                <div>
                  <span class="panel-kicker">Recent Trend</span>
                  <h3>近期分析趋势</h3>
                  <p>按最近完成的视频统计角色数量和片段数量，适合答辩时快速说明分析结果变化。</p>
                </div>

                <div class="trend-badges">
                  <div class="badge-card">
                    <span>可播放记录</span>
                    <strong>{{ playableCount }}</strong>
                  </div>
                  <div class="badge-card muted">
                    <span>标注缺失</span>
                    <strong>{{ missingAnnotatedCount }}</strong>
                  </div>
                </div>
              </div>

              <div v-if="summary.trends.length === 0" class="chart-empty">
                暂无近期趋势数据，等完成更多视频分析后这里会自动更新。
              </div>
              <div v-else ref="chartRef" class="chart"></div>
            </section>

            <section class="panel videos-panel">
              <div class="panel-head compact">
                <div>
                  <span class="panel-kicker">Recent Videos</span>
                  <h3>最近视频记录</h3>
                </div>
              </div>

              <div class="video-list">
                <article
                  v-for="item in summary.recent_videos"
                  :key="item.id"
                  class="video-item"
                  @click="emit('open-history', item)"
                >
                  <div class="video-copy">
                    <strong>{{ item.filename }}</strong>
                    <span>{{ item.character_count }} 个角色 / {{ formatDuration(item.duration) }}</span>
                  </div>

                  <div class="video-side">
                    <span class="video-status" :class="item.has_annotated_video ? 'ready' : 'missing'">
                      {{ item.has_annotated_video ? '可播放' : '标注缺失' }}
                    </span>
                    <span class="video-time">{{ formatDate(item.updated_at) }}</span>
                  </div>
                </article>

                <div v-if="summary.recent_videos.length === 0" class="empty-tip">
                  还没有可展示的视频记录。
                </div>
              </div>
            </section>

            <section class="panel updates-panel">
              <div class="panel-head compact">
                <div>
                  <span class="panel-kicker">Update Feed</span>
                  <h3>最近更新记录</h3>
                </div>
              </div>

              <div class="updates-list">
                <article
                  v-for="item in summary.recent_updates"
                  :key="`${item.kind}-${item.title}-${item.updated_at}`"
                  class="update-item"
                  @click="handleUpdateClick(item)"
                >
                  <span class="update-kind" :class="item.kind">
                    {{ item.kind === 'library' ? '角色库' : '视频分析' }}
                  </span>
                  <div class="update-copy">
                    <strong>{{ item.title }}</strong>
                    <p :title="item.detail">{{ item.detail }}</p>
                  </div>
                  <span class="update-time">{{ formatDate(item.updated_at) }}</span>
                </article>

                <div v-if="summary.recent_updates.length === 0" class="empty-tip">
                  还没有可展示的系统更新记录。
                </div>
              </div>
            </section>
          </div>

          <div class="side-column">
            <section class="panel model-panel">
              <div class="panel-head compact">
                <div>
                  <span class="panel-kicker">Production Stack</span>
                  <h3>当前生产模型</h3>
                </div>
              </div>

              <div class="model-stack">
                <article v-for="card in modelCards" :key="card.key" class="model-card" @click="emit('open-settings-section', 'showcase')">
                  <div class="model-badge">{{ card.badge }}</div>
                  <div class="model-copy">
                    <strong>{{ card.title }}</strong>
                    <span class="model-value">{{ card.value }}</span>
                    <p>{{ card.description }}</p>
                  </div>
                </article>
              </div>
            </section>

            <section class="panel insights-panel">
              <div class="panel-head compact">
                <div>
                  <span class="panel-kicker">Insights</span>
                  <h3>分析洞察预估</h3>
                </div>
              </div>

              <div class="insights-stack">
                <div class="insight-row">
                  <div class="insight-title">角色库识别密度</div>
                  <div class="insight-val highlight">{{ summary.insights.known_recognition_rate.toFixed(1) }}%</div>
                  <p>分析片段中已知角色的识别占比</p>
                </div>
                <div class="insight-row">
                  <div class="insight-title">视频平均时长</div>
                  <div class="insight-val">{{ formatDuration(summary.insights.avg_video_duration) }}</div>
                  <p>库内有效分析样本的平均时间</p>
                </div>
                <div class="insight-row">
                  <div class="insight-title">视频角色饱和度</div>
                  <div class="insight-val">{{ summary.insights.avg_characters_per_video.toFixed(1) }} <span>人/部</span></div>
                  <p>平均每支视频内捕获的独立个体数</p>
                </div>
              </div>
            </section>

            <section class="panel character-panel">
              <div class="panel-head compact">
                <div>
                  <span class="panel-kicker">Library</span>
                  <h3>最近新增角色</h3>
                </div>
              </div>

              <div class="character-list">
                <article
                  v-for="item in summary.recent_characters"
                  :key="item.name"
                  class="character-item"
                  @click="emit('open-settings-section', 'library')"
                >
                  <img
                    v-if="item.portrait_url"
                    :src="toApiUrl(item.portrait_url)"
                    :alt="item.name"
                    class="portrait"
                  />
                  <div v-else class="portrait placeholder">{{ item.name.slice(0, 1) }}</div>

                  <div class="character-copy">
                    <strong>{{ item.name }}</strong>
                    <span>{{ item.sample_count }} 张样本</span>
                    <span>{{ item.portrait_url ? '封面已生成' : '未生成封面' }}</span>
                  </div>
                </article>

                <div v-if="summary.recent_characters.length === 0" class="empty-tip">
                  还没有可展示的角色库角色。
                </div>
              </div>
            </section>
          </div>
        </div>
      </template>
    </el-skeleton>
  </section>
</template>

<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'
import * as echarts from 'echarts'
import { apiGetDashboardSummary, toApiUrl } from '../api/index'

const emit = defineEmits(['open-history', 'open-settings-section'])

const loading = ref(true)
const chartRef = ref(null)
const chartInstance = ref(null)
let resizeTimer = null

const summary = ref({
  stats: [],
  models: {
    yolo_model: '-',
    recognition_model: '-',
    tracker_config: '-',
  },
  trends: [],
  recent_characters: [],
  recent_videos: [],
  recent_updates: [],
  insights: {
    avg_video_duration: 0,
    known_recognition_rate: 0,
    avg_characters_per_video: 0,
    total_appearances: 0,
  },
})

const playableCount = computed(
  () => summary.value.recent_videos.filter((item) => item.has_annotated_video).length,
)

const missingAnnotatedCount = computed(
  () => summary.value.recent_videos.filter((item) => !item.has_annotated_video).length,
)

const heroChips = computed(() => [
  { label: '检测模型', value: summary.value.models.yolo_model || '-' },
  { label: '特征模型', value: summary.value.models.recognition_model || '-' },
  { label: '跟踪配置', value: summary.value.models.tracker_config || '-' },
])

const modelCards = computed(() => [
  {
    key: 'detector',
    badge: 'DET',
    title: 'YOLO 检测',
    value: summary.value.models.yolo_model || '-',
    description: '负责角色候选区域检测、样本裁剪和视频分析预处理。',
  },
  {
    key: 'recognition',
    badge: 'REC',
    title: '特征提取',
    value: summary.value.models.recognition_model || '-',
    description: '负责角色身份匹配、特征向量生成和历史结果比对。',
  },
  {
    key: 'tracker',
    badge: 'TRK',
    title: 'ByteTrack',
    value: summary.value.models.tracker_config || '-',
    description: '负责跨帧 ID 稳定、时序片段拼接和轨迹连续性。',
  },
])

const renderChart = () => {
  if (!chartRef.value || summary.value.trends.length === 0) {
    chartInstance.value?.clear()
    return
  }

  if (!chartInstance.value) {
    chartInstance.value = echarts.init(chartRef.value, null, { renderer: 'canvas' })
  }

  chartInstance.value.setOption(
    {
      backgroundColor: 'transparent',
      animationDuration: 900,
      animationEasing: 'cubicOut',
      animationDurationUpdate: 500,
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
      },
      legend: {
        top: 0,
        textStyle: { color: '#718198' },
      },
      grid: { left: 28, right: 12, top: 48, bottom: 28 },
      xAxis: {
        type: 'category',
        data: summary.value.trends.map((item) => item.label),
        axisLabel: { color: '#718198' },
        axisLine: { lineStyle: { color: 'rgba(135, 157, 201, 0.24)' } },
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: '#718198' },
        splitLine: { lineStyle: { color: 'rgba(135, 157, 201, 0.16)' } },
      },
      series: [
        {
          name: '角色数',
          type: 'line',
          smooth: true,
          symbolSize: 8,
          lineStyle: { width: 3, color: '#4e7cff' },
          itemStyle: { color: '#4e7cff' },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(78, 124, 255, 0.26)' },
              { offset: 1, color: 'rgba(78, 124, 255, 0.03)' },
            ]),
          },
          data: summary.value.trends.map((item) => item.character_count),
        },
        {
          name: '片段数',
          type: 'bar',
          barWidth: 18,
          itemStyle: {
            borderRadius: [8, 8, 0, 0],
            color: '#43c3a5',
          },
          data: summary.value.trends.map((item) => item.total_segments),
        },
      ],
    },
    true,
  )

  chartInstance.value.off('click')
  chartInstance.value.on('click', (params) => {
    const point = summary.value.trends[params.dataIndex]
    if (!point?.video_id) {
      return
    }
    emit('open-history', {
      id: point.video_id,
      filename: point.filename,
      duration: point.duration,
      has_annotated_video: point.has_annotated_video,
    })
  })
}

const scheduleResize = () => {
  if (resizeTimer) {
    clearTimeout(resizeTimer)
  }
  resizeTimer = setTimeout(() => {
    chartInstance.value?.resize()
    resizeTimer = null
  }, 120)
}

const formatDate = (value) => {
  if (!value) return '未知时间'
  try {
    return new Date(value).toLocaleString('zh-CN', {
      month: 'numeric',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    })
  } catch {
    return value
  }
}

const formatDuration = (value) => {
  if (value == null || Number.isNaN(Number(value))) return '时长未知'
  const totalSeconds = Math.max(Math.round(Number(value)), 0)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}

const handleStatClick = (index) => {
  if (index <= 1) {
    emit('open-settings-section', 'library')
    return
  }
  if (summary.value.recent_videos[0]) {
    emit('open-history', summary.value.recent_videos[0])
  }
}

const handleUpdateClick = (item) => {
  if (item.kind === 'library') {
    emit('open-settings-section', 'library')
    return
  }
  if (summary.value.recent_videos[0]) {
    emit('open-history', summary.value.recent_videos[0])
  }
}

const loadSummary = async () => {
  loading.value = true
  try {
    const response = await apiGetDashboardSummary()
    summary.value = {
      stats: response.data?.stats ?? [],
      models: response.data?.models ?? {
        yolo_model: '-',
        recognition_model: '-',
        tracker_config: '-',
      },
      trends: response.data?.trends ?? [],
      recent_characters: response.data?.recent_characters ?? [],
      recent_videos: response.data?.recent_videos ?? [],
      recent_updates: response.data?.recent_updates ?? [],
      insights: response.data?.insights ?? {
        avg_video_duration: 0,
        known_recognition_rate: 0,
        avg_characters_per_video: 0,
        total_appearances: 0,
      },
    }
    await nextTick()
    setTimeout(() => {
      renderChart()
      scheduleResize()
    }, 50)
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  loadSummary()
  window.addEventListener('resize', scheduleResize)
})

onUnmounted(() => {
  if (resizeTimer) {
    clearTimeout(resizeTimer)
  }
  window.removeEventListener('resize', scheduleResize)
  chartInstance.value?.dispose()
  chartInstance.value = null
})
</script>

<style scoped>
.dashboard-page {
  min-height: 0;
  padding: 24px 32px 40px;
  display: flex;
  flex-direction: column;
  gap: 24px;
  overflow-y: auto;
}

.dashboard-hero {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 18px;
  padding: 0 0 16px 0;
  border-bottom: 1px solid var(--border-color);
  margin-bottom: 8px;
}

.hero-kicker,
.panel-kicker {
  display: inline-block;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-secondary);
}

.hero-copy h1,
.panel-head h3 {
  margin: 6px 0 8px;
  color: var(--text-primary);
}

.hero-copy h1 {
  font-size: 28px;
  font-weight: 600;
  line-height: 1.2;
}

.hero-copy p,
.panel-head p {
  margin: 0;
  color: var(--text-secondary);
  font-size: 14px;
  line-height: 1.6;
}

.hero-actions {
  display: flex;
  flex-wrap: wrap;
  align-content: flex-start;
  justify-content: flex-end;
  gap: 12px;
}

.hero-chip {
  min-width: 120px;
  padding: 10px 14px;
  border-radius: 8px;
  background: #ffffff;
  border: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.hero-chip span {
  font-size: 12px;
  color: #7584a1;
}

.hero-chip strong {
  font-size: 14px;
  color: #22304a;
  word-break: break-all;
}

.refresh-btn {
  height: 40px;
  border-radius: 8px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}

.metric-card,
.panel {
  background: #ffffff;
  border: 1px solid var(--border-color);
  border-radius: 12px;
  box-shadow: var(--shadow-light);
}

.metric-card.interactive,
.model-card,
.character-item,
.video-item,
.update-item {
  cursor: pointer;
}

.metric-card {
  padding: 22px 24px;
}

.metric-card.interactive:hover,
.model-card:hover,
.character-item:hover,
.video-item:hover,
.update-item:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-medium);
}

.metric-label {
  font-size: 13px;
  color: var(--text-secondary);
  font-weight: 500;
}

.metric-value {
  margin-top: 10px;
  font-size: 32px;
  font-weight: 600;
  color: var(--text-primary);
}

.metric-hint {
  margin-top: 6px;
  color: var(--text-secondary);
  font-size: 12px;
}

.dashboard-grid {
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(320px, 1fr);
  gap: 16px;
  align-items: start;
}

.panel {
  padding: 24px;
}

.panel-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 18px;
}

.panel-head.compact {
  margin-bottom: 16px;
}

.trend-panel {
  min-height: 430px;
}

.trend-badges {
  display: flex;
  gap: 10px;
}

.badge-card {
  min-width: 100px;
  padding: 10px 12px;
  border-radius: 8px;
  background: #f8fafc;
  border: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.badge-card span {
  font-size: 12px;
  color: #718198;
}

.badge-card strong {
  font-size: 22px;
  color: #23314d;
}

.badge-card.muted {
  background: #fff6f6;
}

.chart {
  width: 100%;
  height: 320px;
}

.chart-empty,
.empty-tip {
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  color: var(--text-secondary);
  border-radius: 8px;
  background: #fdfdfd;
  border: 1px dashed var(--border-color);
}

.chart-empty {
  min-height: 320px;
  padding: 24px;
}

.main-column {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.side-column {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.model-stack,
.character-list,
.video-list,
.updates-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.model-card,
.character-item,
.video-item,
.update-item {
  border-radius: 8px;
  background: #ffffff;
  border: 1px solid var(--border-color);
}

.model-card {
  padding: 16px;
  display: grid;
  grid-template-columns: 52px minmax(0, 1fr);
  gap: 14px;
}

.model-badge {
  width: 48px;
  height: 48px;
  border-radius: 8px;
  background: #f1f5f9;
  color: var(--text-secondary);
  display: grid;
  place-items: center;
  font-weight: 600;
  font-size: 14px;
}

.model-copy {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.model-copy strong { color: var(--text-primary); }

.model-value {
  color: #4d6388;
  word-break: break-all;
}

.model-copy p {
  margin: 0;
  font-size: 13px;
  color: #7d8aa2;
  line-height: 1.6;
}

.character-item {
  padding: 14px;
  display: flex;
  align-items: center;
  gap: 12px;
}

.portrait {
  width: 48px;
  height: 48px;
  border-radius: 8px;
  object-fit: cover;
  background: #f1f5f9;
  flex-shrink: 0;
}

.portrait.placeholder {
  display: grid;
  place-items: center;
  font-size: 22px;
  font-weight: 700;
  color: #5273bf;
}

.character-copy,
.video-copy,
.update-copy {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.character-copy strong,
.video-copy strong,
.update-copy strong { color: var(--text-primary); }

.character-copy span,
.video-copy span,
.video-time,
.update-time {
  font-size: 13px;
  color: var(--text-secondary);
}

.update-copy p {
  margin: 0;
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
}

.video-item,
.update-item {
  padding: 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.video-name {
  word-break: break-all;
}

.video-side {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 6px;
  flex-shrink: 0;
}

.video-status,
.update-kind {
  padding: 5px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
}

.video-status.ready {
  background: rgba(69, 201, 154, 0.16);
  color: #159565;
}

.video-status.missing {
  background: rgba(255, 110, 122, 0.13);
  color: #d65a67;
}

.update-kind.library { background: #f1f5f9; color: var(--text-secondary); border: 1px solid var(--border-color); }
.update-kind.video { background: #f1f5f9; color: var(--text-secondary); border: 1px solid var(--border-color); }

.updates-panel {
}

.updates-list {
  display: grid;
  grid-template-columns: 1fr;
  gap: 12px;
}

.insights-stack {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.insight-row {
  background: #f8fafc;
  border-radius: 8px;
  padding: 16px;
  border: 1px solid var(--border-color);
}

.insight-title {
  font-size: 13px;
  color: var(--text-secondary);
  margin-bottom: 8px;
}

.insight-val {
  font-size: 26px;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 6px;
}

.insight-val.highlight {
  color: var(--accent-primary);
}

.insight-val span {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-secondary);
}

.insight-row p {
  margin: 0;
  font-size: 12px;
  color: var(--text-secondary);
}

.update-item {
  align-items: flex-start;
}

@media (max-width: 1280px) {
  .dashboard-hero { flex-direction: column; align-items: flex-start; }
  .dashboard-grid, .updates-list { grid-template-columns: 1fr; }
  .stats-grid { grid-template-columns: repeat(2, 1fr); }
  .hero-actions { justify-content: flex-start; margin-top: 10px; }
}

@media (max-width: 768px) {
  .dashboard-page {
    padding: 18px 16px 28px;
  }

  .dashboard-hero,
  .panel,
  .metric-card {
    padding: 18px;
    border-radius: 20px;
  }

  .hero-copy h1 {
    font-size: 30px;
  }

  .metric-value {
    font-size: 36px;
  }

  .trend-badges {
    width: 100%;
    flex-direction: column;
  }

  .video-item,
  .update-item {
    flex-direction: column;
    align-items: flex-start;
  }

  .video-side {
    align-items: flex-start;
  }
}
</style>
