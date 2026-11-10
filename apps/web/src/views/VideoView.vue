<template>
  <main class="main-content">
    <div class="content-area">
      <div class="welcome-box" 
           @click="$refs.fileInput.click()"
           @drop="handleDrop" 
           @dragover.prevent
           v-if="status === 'idle'">
        <el-icon :size="60" color="#a1a5b7"><UploadFilled /></el-icon>
        <h2>将动漫视频拖拽至此，或点击上传</h2>
        <p>支持 mp4、avi 格式，建议视频长度小于 15 分钟</p>
        <input type="file" ref="fileInput" @change="handleFileChange" accept="video/mp4" style="display: none;" />
        <el-button type="primary" size="large" class="upload-btn" @click.stop="$refs.fileInput.click()">选择视频文件</el-button>
      </div>

      <div class="progress-box" v-if="(status === 'uploading' || status === 'processing') && !isViewingHistory">
        <h2>{{ status === 'uploading' ? '正在上传到服务器...' : 'AI 正在逐帧提取与跟踪...' }}</h2>
        <el-progress :percentage="progress" :stroke-width="24" striped striped-flow :duration="10" color="#3b82f6" />
        <p class="progress-tip">处理耗时：{{ processingTime }} 秒</p>
      </div>
      <div class="progress-box" v-if="status === 'loading_history'">
        <h2>正在从数据库调取历史档案...</h2>
      </div>

      <div class="result-workspace" v-show="status === 'completed'">
        
        <div class="stats-panel">
          <el-row :gutter="20">
            <el-col :span="6">
              <el-card class="stat-card">
                <div class="stat-item">
                  <div class="stat-label">过滤后角色数</div>
                  <div class="stat-value">{{ Object.keys(filteredTimelineData).length }}</div>
                </div>
              </el-card>
            </el-col>
            <el-col :span="6">
              <el-card class="stat-card">
                <div class="stat-item">
                  <div class="stat-label">已知角色</div>
                  <div class="stat-value" style="color: var(--accent-secondary);">{{ knownCount }}</div>
                </div>
              </el-card>
            </el-col>
            <el-col :span="6">
              <el-card class="stat-card">
                <div class="stat-item">
                  <div class="stat-label">陌生人</div>
                  <div class="stat-value" style="color: #f59e0b;">{{ strangerCount }}</div>
                </div>
              </el-card>
            </el-col>
            <el-col :span="6">
              <el-card class="stat-card">
                <div class="stat-item">
                  <div class="stat-label">{{ isViewingHistory ? '视频总时长' : 'AI 处理耗时' }}</div>
                  <div class="stat-value">
                    {{ isViewingHistory ? formatDurationLabel(timelineMeta.duration_sec) : `${processingTime}s` }}
                  </div>
                </div>
              </el-card>
            </el-col>
          </el-row>
        </div>

        <el-alert
          v-if="isViewingHistory && !historyVideoPlayable"
          :title="historyVideoMessage || '该历史记录缺少标注视频文件，当前无法播放视频'"
          type="warning"
          show-icon
          :closable="false"
          class="history-video-alert"
        />

        <div class="video-container" :class="{ disabled: isViewingHistory && !historyVideoPlayable }">
          <video
            v-if="annotatedVideoUrl || localVideoUrl"
            ref="videoRef"
            :src="annotatedVideoUrl || localVideoUrl"
            controls
            class="anime-video"
          ></video>
          <div v-else class="video-empty-state">标注视频缺失，当前仅展示时间轴与图表数据</div>
        </div>
        
        <div class="charts-container">
          <div class="chart-row">
            <div class="chart-box" style="flex: 1;">
              <div class="chart-header">
                <h3 class="chart-title"><el-icon><Histogram /></el-icon> 角色出场时间轴（点击色块可跳转视频）</h3>
                
                <div class="filter-control">
                  <span class="filter-label">过滤路人（出场少于 {{ minDurationFilter }}s）</span>
                  <el-slider 
                    v-model="minDurationFilter" 
                    :max="15" 
                    :step="0.5" 
                    :show-tooltip="false"
                    @change="onFilterChange"
                    style="width: 160px; margin-left: 12px;" 
                  />
                </div>
              </div>
              <div id="gantt-chart" style="width: 100%; height: 350px;"></div>
              <div class="pagination" v-if="characters.length > itemsPerPage">
                <button @click="prevPage" :disabled="currentPage === 0">上一页</button>
                <span>第 {{ currentPage + 1 }} 页 / 共 {{ totalPages }} 页</span>
                <button @click="nextPage" :disabled="currentPage >= totalPages - 1">下一页</button>
              </div>
            </div>
          </div>
          
          <div class="chart-row">
            <div class="chart-box">
              <div class="chart-header">
                <h3 class="chart-title"><el-icon><PieChart /></el-icon> 出场时长占比</h3>
                <el-radio-group v-model="pieMode" size="small" @change="renderPieChartStable">
                  <el-radio-button value="simplified">简略</el-radio-button>
                  <el-radio-button value="detailed">详细</el-radio-button>
                </el-radio-group>
              </div>
              <div id="pie-chart" style="width: 100%; height: 300px;"></div>
            </div>
            <div class="chart-box">
              <h3 class="chart-title"><el-icon><DataLine /></el-icon> 角色出场总时长排行</h3>
              <div id="bar-chart" style="width: 100%; height: 300px;"></div>
            </div>
          </div>

          <div class="chart-row">
            <div class="chart-box" style="flex: 1;">
              <h3 class="chart-title"><el-icon><Grid /></el-icon> 角色同框热度矩阵（交集时长）</h3>
              <div id="heatmap-chart" style="width: 100%; height: 400px;"></div>
            </div>
          </div>

        </div>
      </div>
    </div>

    <div class="floating-progress" v-if="isBgTaskRunning && isViewingHistory">
      <div class="fp-header">
        <span>AI 正在后台解析新视频...</span>
        <span>{{ processingTime }}s</span>
      </div>
      <el-progress :percentage="progress" :stroke-width="12" striped striped-flow color="var(--accent-secondary)" />
    </div>

  </main>
</template>

<script setup>
import { ref, nextTick, computed, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import { UploadFilled, Histogram, PieChart, DataLine, Grid } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { apiUploadVideo, apiGetVideoStatus, apiGetTimeline, apiRenameStranger } from '../api/index'

const props = defineProps({
  analysisModelSelection: {
    type: Object,
    default: () => ({
      detection_model_path: '',
      recognition_model_path: '',
      tracker_config_path: '',
    }),
  },
  dataExportEnabled: {
    type: Boolean,
    default: false,
  },
})

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  import.meta.env.VITE_API_URL ||
  'http://localhost:8000'

const status = ref('idle')
const isViewingHistory = ref(false)
const timelineData = ref({}) // 鍘熷鍏ㄩ噺鏁版嵁
const timelineMeta = ref({ duration_sec: 0, fps: 0 })
const annotatedVideoUrl = ref('')
const localVideoUrl = ref('')
const videoRef = ref(null)
const historyVideoPlayable = ref(true)
const historyVideoMessage = ref('')

const isBgTaskRunning = ref(false)
const bgVideoId = ref(null)
const progress = ref(0)
const processingTime = ref(0)
let pollInterval = null

const pieMode = ref('simplified')
const currentPage = ref(0)
const itemsPerPage = 5
let resizeHandler = null
let renderFrame = null
let resizeTimer = null
let layoutRefreshTimer = null
const chartInstances = {
  gantt: null,
  pie: null,
  bar: null,
  heatmap: null,
}

const nameDict = {
  Aquamarine_Hoshino: '阿库亚',
  Hoshino_Ruby: '星野露比',
  Hoshino_Ai: '星野爱',
}

const translateName = (name) => {
  if (!name) return ''
  if (name.startsWith('Stranger_')) {
    return name.replace('Stranger_', '陌生角色_')
  }
  return nameDict[name] || name
}

const getHighlightName = (name, isMVP = false) => {
  const displayName = translateName(name)
  return isMVP ? `TOP1 ${displayName}` : displayName
}

// 馃殌 鏂板锛氳矾浜鸿繃婊ら槇鍊?(榛樿涓?0 绉掞紝鍗充笉瑕嗙洊)
const formatDurationLabel = (seconds) => {
  if (seconds == null || Number.isNaN(Number(seconds))) {
    return '时长未知'
  }
  const totalSeconds = Math.max(Math.round(Number(seconds)), 0)
  const minutes = Math.floor(totalSeconds / 60)
  const remainder = totalSeconds % 60
  if (minutes <= 0) {
    return `${remainder}s`
  }
  return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
}

const minDurationFilter = ref(0)

// 馃殌 鏍稿績璁＄畻灞炴€э細缁忚繃婊ゅ櫒绛涙礂鍚庣殑骞插噣鏁版嵁
const filteredTimelineData = computed(() => {
  const result = {}
  for (const [name, info] of Object.entries(timelineData.value)) {
    const total = info.segments.reduce((acc, seg) => acc + (seg.end - seg.start), 0)
    // 鍙湁鎬绘椂闀垮ぇ浜庣瓑浜庨槇鍊肩殑瑙掕壊鎵嶄細琚暀涓?
    if (total >= minDurationFilter.value) {
      result[name] = info
    }
  }
  return result
})

// 鎵€鏈夌殑缁熻閮藉熀浜?filteredTimelineData (绾噣鏁版嵁) 璁＄畻
const knownCount = computed(() => Object.values(filteredTimelineData.value).filter(d => d.type === 'known').length)
const strangerCount = computed(() => Object.values(filteredTimelineData.value).filter(d => d.type === 'stranger').length)
const totalPages = computed(() => Math.ceil(Object.keys(filteredTimelineData.value).length / itemsPerPage))

const characterDurations = computed(() => {
  const data = []
  for (const [name, info] of Object.entries(filteredTimelineData.value)) {
    let totalDuration = 0
    info.segments.forEach(seg => { totalDuration += (seg.end - seg.start) })
    data.push({ name, total: Number(totalDuration.toFixed(2)), type: info.type })
  }
  // 鎸夋椂闀块檷搴忔帓鍒?
  data.sort((a, b) => b.total - a.total)
  
  // 馃殌 鏅鸿兘 C 浣嶏細缁欑涓€鍚嶆墦涓?MVP 鏍囪
  if (data.length > 0) {
    data[0].isMVP = true
  }
  return data
})

const characters = computed(() => {
  const known = Object.entries(filteredTimelineData.value).filter(([_, info]) => info.type === 'known').sort(([a], [b]) => a.localeCompare(b))
  const stranger = Object.entries(filteredTimelineData.value).filter(([_, info]) => info.type === 'stranger').sort(([keyA, aInfo], [keyB, bInfo]) => aInfo.segments[0].start - bInfo.segments[0].start)
  return [...known, ...stranger]
})

const timelineMaxSecond = computed(() => {
  let maxEnd = Number(timelineMeta.value?.duration_sec || 0)
  for (const info of Object.values(filteredTimelineData.value)) {
    for (const segment of info.segments || []) {
      maxEnd = Math.max(maxEnd, Number(segment.end || 0))
    }
  }
  return Math.max(1, Math.ceil(maxEnd))
})

// --- 鏂规硶 ---
const releaseLocalUrl = () => { if (localVideoUrl.value) URL.revokeObjectURL(localVideoUrl.value) }
const resetHistoryPlaybackState = () => {
  historyVideoPlayable.value = true
  historyVideoMessage.value = ''
}

const getChartInstance = (key, domId) => {
  const dom = document.getElementById(domId)
  if (!dom) return null
  if (!chartInstances[key]) {
    chartInstances[key] = echarts.init(dom)
  }
  return chartInstances[key]
}

const clearChart = (key) => {
  chartInstances[key]?.clear()
}

const clearAllCharts = () => {
  Object.keys(chartInstances).forEach((key) => clearChart(key))
}

const resizeAllCharts = () => {
  if (resizeTimer) {
    clearTimeout(resizeTimer)
  }
  resizeTimer = setTimeout(() => {
    Object.values(chartInstances).forEach((instance) => instance?.resize())
    resizeTimer = null
  }, 80)
}

const scheduleRender = (scope = 'all') => {
  if (renderFrame) {
    cancelAnimationFrame(renderFrame)
  }
  renderFrame = requestAnimationFrame(() => {
    if (scope === 'gantt') {
      renderGanttChart()
    } else {
      renderAllCharts()
    }
    renderFrame = null
  })
}

const refreshLayout = () => {
  nextTick(() => {
    resizeAllCharts()
    scheduleRender()
    if (layoutRefreshTimer) {
      clearTimeout(layoutRefreshTimer)
    }
    layoutRefreshTimer = setTimeout(() => {
      resizeAllCharts()
      layoutRefreshTimer = null
    }, 140)
  })
}

const handleRenameStranger = async (strangerName) => {
  try {
    const { value: newName } = await ElMessageBox.prompt(
      `请为 ${translateName(strangerName)} 输入真实角色名（例如 Hoshino_Ruby）：`,
      '陌生人转正 / 特征融合',
      {
        confirmButtonText: '确认转正',
        cancelButtonText: '取消',
        inputPattern: /^[a-zA-Z0-9_]+$/,
        inputErrorMessage: '名字只能包含英文、数字和下划线。',
        inputPlaceholder: '例如：Hoshino_Ruby',
      }
    )

    if (newName) {
      const activeRecognitionModelPath =
        timelineMeta.value?.model_selection?.recognition_model_path ||
        props.analysisModelSelection?.recognition_model_path ||
        null
      const res = await apiRenameStranger(strangerName, newName, activeRecognitionModelPath)
      const isFused = res.data.fused
      
      ElMessage.success(
        isFused 
          ? `已融合：${translateName(strangerName)} 的特征已并入 ${translateName(newName)}`
          : `已转正：${translateName(strangerName)} 已成为新角色 ${translateName(newName)}`
      )
      
      if (timelineData.value[strangerName]) {
        const strangerInfo = timelineData.value[strangerName]
        strangerInfo.type = 'known'
        
        if (timelineData.value[newName]) {
          timelineData.value[newName].segments.push(...strangerInfo.segments)
          timelineData.value[newName].segments.sort((a, b) => a.start - b.start)
        } else {
          timelineData.value[newName] = strangerInfo
        }
        
        delete timelineData.value[strangerName]
        refreshLayout()
      }
    }
  } catch (error) {
    if (error !== 'cancel') {
      console.error('转正失败:', error)
      ElMessage.error(`转正失败：${error.response?.data?.detail || error.message}`)
    }
  }
}

const resetForNewTask = () => {
  isViewingHistory.value = false
  resetHistoryPlaybackState()
  timelineMeta.value = { duration_sec: 0, fps: 0 }
  if (isBgTaskRunning.value) { status.value = 'processing' } 
  else { status.value = 'idle'; annotatedVideoUrl.value = ''; timelineData.value = {}; minDurationFilter.value = 0; }
  currentPage.value = 0
  clearAllCharts()
}

const loadHistoryData = async (historyRecord) => {
  const vid = typeof historyRecord === 'object' ? historyRecord?.id : historyRecord
  status.value = 'loading_history'
  isViewingHistory.value = true
  minDurationFilter.value = 0
  resetHistoryPlaybackState()
  if (historyRecord && typeof historyRecord === 'object' && historyRecord.has_annotated_video === false) {
    historyVideoPlayable.value = false
    historyVideoMessage.value = '该历史记录缺少标注视频文件，当前无法播放视频'
    annotatedVideoUrl.value = ''
  } else {
    annotatedVideoUrl.value = `${API_BASE_URL}/api/videos/${vid}/download`
  }
  
  try {
    const response = await apiGetTimeline(vid)
    timelineData.value = response.data.timeline
    timelineMeta.value = response.data.video_info || { duration_sec: 0, fps: 0 }
    status.value = 'completed'
    refreshLayout()
  } catch (error) {
    console.error('拉取历史失败', error)
    alert('档案调取失败')
    resetForNewTask()
  }
}

defineExpose({ resetForNewTask, loadHistoryData, refreshLayout })

const handleFileChange = (event) => {
  const file = event.target.files[0]
  if (!file) return
  releaseLocalUrl()
  localVideoUrl.value = URL.createObjectURL(file)
  uploadVideo(file)
}

const handleDrop = (e) => {
  e.preventDefault()
  const file = e.dataTransfer.files[0]
  if (file && file.type.startsWith('video/')) {
    releaseLocalUrl()
    localVideoUrl.value = URL.createObjectURL(file)
    uploadVideo(file)
  }
}

const uploadVideo = async (file) => {
  status.value = 'uploading'
  isViewingHistory.value = false 
  resetHistoryPlaybackState()
  isBgTaskRunning.value = true
  progress.value = 0
  processingTime.value = 0
  minDurationFilter.value = 0
  annotatedVideoUrl.value = ''
  
  try {
    const response = await apiUploadVideo(file, props.analysisModelSelection, props.dataExportEnabled)
    bgVideoId.value = response.data.video_id
    status.value = 'processing'
    startPolling()
  } catch (error) {
    status.value = 'idle'
    isBgTaskRunning.value = false
    alert('上传失败')
  }
}

const startPolling = () => {
  const startTime = Date.now()
  pollInterval = setInterval(async () => {
    try {
      const response = await apiGetVideoStatus(bgVideoId.value)
      const data = response.data
      progress.value = data.progress || 0
      processingTime.value = Math.floor((Date.now() - startTime) / 1000)
      
      if (data.status === 'completed') {
        clearInterval(pollInterval)
        pollInterval = null
        isBgTaskRunning.value = false
        
        isViewingHistory.value = false
        resetHistoryPlaybackState()
        annotatedVideoUrl.value = `${API_BASE_URL}/api/videos/${bgVideoId.value}/download`
        
        const tlRes = await apiGetTimeline(bgVideoId.value)
        timelineData.value = tlRes.data.timeline
        timelineMeta.value = tlRes.data.video_info || { duration_sec: 0, fps: 0 }
        status.value = 'completed'
        
        refreshLayout()
        
      } else if (data.status === 'failed') {
        clearInterval(pollInterval)
        pollInterval = null
        isBgTaskRunning.value = false
        if(!isViewingHistory.value) { status.value = 'idle' }
        alert('AI 处理失败')
      }
    } catch (error) {
      console.error('轮询失败', error)
    }
  }, 2000)
}

// ======================= 鍥捐〃娓叉煋鏍稿績 =======================
const renderAllCharts = () => { 
  if (!characters.value.length) {
    clearAllCharts()
    return
  }
  renderGanttChart(); 
  renderPieChartStable(); 
  renderBarChartStable(); 
  renderHeatmapChart(); // 娓叉煋鏂扮殑鐭╅樀鍥?
}

const onFilterChange = () => {
  currentPage.value = 0
  refreshLayout()
}

const renderGanttChart = () => {
  const myChart = getChartInstance('gantt', 'gantt-chart'); if (!myChart) return
  
  const paginatedData = characters.value.slice(currentPage.value * itemsPerPage, (currentPage.value + 1) * itemsPerPage)
  if (!paginatedData.length) {
    myChart.clear()
    return
  }
  const chartData = []
  const colors = { 'known': '#10b981', 'stranger': '#f59e0b' }
  
  paginatedData.forEach(([charName, info], index) => {
    const isMVP = characterDurations.value.length > 0 && characterDurations.value[0].name === charName
    const displayName = getHighlightName(charName, isMVP)

    info.segments.forEach(seg => {
      chartData.push({
        name: displayName,
        realName: charName,
        value: [index, seg.start, seg.end, seg.end - seg.start],
        itemStyle: { color: colors[info.type] || '#3b82f6', borderRadius: 4 }
      })
    })
  })
  
  const option = {
    tooltip: { formatter: params => `${params.marker} <b>${params.data.name}</b><br/>出场: ${params.value[1].toFixed(2)}s ~ ${params.value[2].toFixed(2)}s` },
    dataZoom: [{ type: 'slider', xAxisIndex: 0, start: 0, end: 100 }, { type: 'inside', xAxisIndex: 0 }],
    grid: { left: '3%', right: '8%', bottom: '15%', top: '5%', containLabel: true },
    xAxis: { type: 'value', position: 'top', min: 0, max: timelineMaxSecond.value, axisLabel: { formatter: '{value}s' } },
    yAxis: { 
      type: 'category', 
      data: paginatedData.map(([name]) => {
         const isMVP = characterDurations.value.length > 0 && characterDurations.value[0].name === name
         return getHighlightName(name, isMVP)
      }),
      inverse: true, axisTick: { show: false }, axisLine: { show: false } 
    },
    series: [{
      type: 'custom',
      renderItem: (params, api) => {
        const categoryIndex = api.value(0)
        const start = api.coord([api.value(1), categoryIndex]); const end = api.coord([api.value(2), categoryIndex])
        const height = api.size([0, 1])[1] * 0.5
        const rectShape = echarts.graphic.clipRectByRect( 
          { x: start[0], y: start[1] - height/2, width: end[0]-start[0], height }, 
          { x: params.coordSys.x, y: params.coordSys.y, width: params.coordSys.width, height: params.coordSys.height }
        )
        return rectShape && { type: 'rect', shape: rectShape, style: api.style() }
      },
      encode: { x: [1, 2], y: 0 }, data: chartData
    }]
  }
  myChart.setOption(option, true)
  myChart.off('click')
  myChart.on('click', (params) => { if (videoRef.value) { videoRef.value.currentTime = params.value[1]; videoRef.value.play() } })
}

const renderPieChart = () => {
  const myChart = getChartInstance('pie', 'pie-chart'); if (!myChart) return
  let pieData = []
  
  if (pieMode.value === 'simplified') {
    let knownTotal = 0, strangerTotal = 0
    characterDurations.value.forEach(d => { if(d.type === 'known') knownTotal += d.total; else strangerTotal += d.total })
    pieData = [
      { value: knownTotal.toFixed(2), name: '已知角色总时长', itemStyle: { color: '#10b981' } },
      { value: strangerTotal.toFixed(2), name: '陌生人总时长', itemStyle: { color: '#f59e0b' } },
    ]
  } else {
    pieData = characterDurations.value.slice(0, 8).map(d => ({ 
      value: d.total, 
      name: d.isMVP ? `馃憫 ${d.name}` : d.name, 
      itemStyle: { color: d.type === 'known' ? '#10b981' : '#f59e0b' } 
    }))
  }
  myChart.setOption({
    tooltip: { trigger: 'item', formatter: '{a} <br/>{b}: {c} 秒 ({d}%)' },
    series: [{
      name: '出场时长',
      type: 'pie',
      radius: ['40%', '70%'],
      avoidLabelOverlap: false,
      label: { show: pieMode.value === 'detailed', position: 'outside' },
      data: pieData,
      animationDuration: 700,
    }],
  })
}

const renderBarChart = () => {
  const myChart = getChartInstance('bar', 'bar-chart')
  if (!myChart) return
  
  const top10 = characterDurations.value.slice(0, 10)
  if (!top10.length) {
    myChart.clear()
    return
  }
  
  myChart.setOption({
    tooltip: { 
      trigger: 'axis', 
      formatter: '{b} <br/>总时长: {c} 秒<br/><span style="color:#999;font-size:12px">点击陌生人可转正</span>' 
    },
    xAxis: { 
      type: 'category', 
      data: top10.map(d => d.isMVP ? `👑 ${translateName(d.name)}` : translateName(d.name)), 
      axisLabel: { interval: 0, rotate: 30 } 
    },
    yAxis: { type: 'value', name: '时长（秒）' },
    grid: { left: '3%', right: '4%', bottom: '15%', containLabel: true },
    series: [{ 
      data: top10.map(d => ({ 
        value: d.total, 
        itemStyle: { 
          color: d.type === 'known' ? '#10b981' : '#f59e0b', 
          borderRadius: [4, 4, 0, 0] 
        } 
      })), 
      type: 'bar', 
      barWidth: '40%',
      cursor: 'pointer',
      animationDuration: 750,
    }]
  })

  myChart.off('click')
  myChart.on('click', (params) => {
    const originalData = characterDurations.value.find(c => 
      (c.isMVP ? `👑 ${translateName(c.name)}` : translateName(c.name)) === params.name
    )
    
    if (!originalData) return
    
    const originalName = originalData.name
    if (originalName.startsWith('Stranger_')) {
      handleRenameStranger(originalName)
    } else {
      ElMessage.info(`${translateName(originalName)} 已是已知角色，无需转正`)
    }
  })
  
}




const renderPieChartStable = () => {
  const myChart = getChartInstance('pie', 'pie-chart')
  if (!myChart) return

  let pieData = []
  if (pieMode.value === 'simplified') {
    let knownTotal = 0
    let strangerTotal = 0
    characterDurations.value.forEach((item) => {
      if (item.type === 'known') knownTotal += item.total
      else strangerTotal += item.total
    })
    pieData = [
      { value: Number(knownTotal.toFixed(2)), name: '已知角色总时长', tooltipName: '已知角色总时长', itemStyle: { color: '#10b981' } },
      { value: Number(strangerTotal.toFixed(2)), name: '陌生人总时长', tooltipName: '陌生人总时长', itemStyle: { color: '#f59e0b' } },
    ]
  } else {
    pieData = characterDurations.value.slice(0, 8).map((item) => ({
      value: item.total,
      name: item.name,
      displayName: getHighlightName(item.name, item.isMVP),
      tooltipName: getHighlightName(item.name, item.isMVP),
      itemStyle: { color: item.type === 'known' ? '#10b981' : '#f59e0b' },
    }))
  }

  const isDetailed = pieMode.value === 'detailed'
  myChart.setOption({
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        const label = params.data?.tooltipName || params.name
        return `${params.seriesName}<br/>${label}: ${Number(params.value).toFixed(2)} 秒 (${params.percent}%)`
      },
    },
    series: [{
      name: '出场时长',
      type: 'pie',
      radius: ['40%', '70%'],
      avoidLabelOverlap: true,
      minShowLabelAngle: isDetailed ? 4 : 0,
      label: isDetailed ? {
        show: true,
        position: 'outside',
        formatter: ({ data }) => data.displayName || translateName(data.name),
        color: '#303133',
        fontSize: 12,
        width: 132,
        overflow: 'truncate',
      } : { show: false },
      labelLine: isDetailed ? {
        show: true,
        length: 18,
        length2: 12,
        smooth: 0.2,
        lineStyle: { color: 'rgba(110, 110, 128, 0.6)' },
      } : { show: false },
      labelLayout: isDetailed ? { moveOverlap: 'shiftY', hideOverlap: false } : undefined,
      emphasis: {
        scale: true,
        scaleSize: 8,
        label: isDetailed ? {
          show: true,
          fontWeight: 700,
          formatter: ({ data }) => data.displayName || translateName(data.name),
          width: 140,
          overflow: 'truncate',
        } : { show: false },
      },
      data: pieData,
      animationDuration: 700,
    }],
  }, true)
}

const renderBarChartStable = () => {
  const myChart = getChartInstance('bar', 'bar-chart')
  if (!myChart) return

  const top10 = characterDurations.value.slice(0, 10)
  if (!top10.length) {
    myChart.clear()
    return
  }

  myChart.setOption({
    tooltip: {
      trigger: 'axis',
      formatter: '{b} <br/>总时长: {c} 秒<br/><span style="color:#999;font-size:12px">点击陌生人可转正</span>',
    },
    xAxis: {
      type: 'category',
      data: top10.map((item) => getHighlightName(item.name, item.isMVP)),
      axisLabel: { interval: 0, rotate: 30 },
    },
    yAxis: { type: 'value', name: '时长（秒）' },
    grid: { left: '3%', right: '4%', bottom: '15%', containLabel: true },
    series: [{
      data: top10.map((item) => ({
        value: item.total,
        rawName: item.name,
        itemStyle: {
          color: item.type === 'known' ? '#10b981' : '#f59e0b',
          borderRadius: [4, 4, 0, 0],
        },
      })),
      type: 'bar',
      barWidth: '40%',
      cursor: 'pointer',
      animationDuration: 750,
    }],
  }, true)

  myChart.off('click')
  myChart.on('click', (params) => {
    const originalData = top10[params.dataIndex]
    if (!originalData) return

    const originalName = originalData.name
    if (originalName.startsWith('Stranger_')) {
      handleRenameStranger(originalName)
    } else {
      ElMessage.info(`${translateName(originalName)} 已是已知角色，无需转正`)
    }
  })
}

const renderHeatmapChart = () => {
  const myChart = getChartInstance('heatmap', 'heatmap-chart'); if (!myChart) return
  if (!characterDurations.value.length) {
    myChart.clear()
    return
  }

  const candidates = characterDurations.value.slice(0, 15).map(c => c.name)
  
  const overlapMap = {}
  const validCharsSet = new Set()
  let maxOverlap = 0

  for (let i = 0; i < candidates.length; i++) {
    for (let j = 0; j < candidates.length; j++) {
      if (i === j) continue
      
      let overlap = 0
      const segsA = filteredTimelineData.value[candidates[i]].segments
      const segsB = filteredTimelineData.value[candidates[j]].segments
      
      for (const a of segsA) {
        for (const b of segsB) {
          const start = Math.max(a.start, b.start)
          const end = Math.min(a.end, b.end)
          if (start < end) overlap += (end - start)
        }
      }
      
      const overlapFixed = Number(overlap.toFixed(2))
      overlapMap[`${candidates[i]}|${candidates[j]}`] = overlapFixed
      
      if (overlapFixed > 0) {
        validCharsSet.add(candidates[i])
        validCharsSet.add(candidates[j])
        if (overlapFixed > maxOverlap) maxOverlap = overlapFixed
      }
    }
  }

  const topChars = candidates.filter(c => validCharsSet.has(c))
  
  if (topChars.length === 0) {
    myChart.clear()
    return
  }

  const matrixData = []
  for (let i = 0; i < topChars.length; i++) {
    for (let j = 0; j < topChars.length; j++) {
      if (i === j) {
        matrixData.push([j, i, '-'])
        continue
      }
      const overlapVal = overlapMap[`${topChars[i]}|${topChars[j]}`]
      matrixData.push([j, i, overlapVal > 0 ? overlapVal : '-']) 
    }
  }

  const option = {
    tooltip: { 
      position: 'top', 
      formatter: p => p.data[2] === '-' ? '无交集' : `${translateName(topChars[p.data[1]])} & ${translateName(topChars[p.data[0]])}<br/>同框: ${p.data[2]} 秒`,
    },
    grid: { left: '12%', right: '15%', top: '10%', bottom: '20%', containLabel: true },
    xAxis: { 
      type: 'category', 
      data: topChars.map((name) => translateName(name)), 
      splitArea: { show: true }, 
      axisLabel: { interval: 0, rotate: 35, color: '#000000' } 
    },
    yAxis: { 
      type: 'category', 
      data: topChars.map((name) => translateName(name)), 
      splitArea: { show: true }, 
      axisLabel: { color: '#000000' } 
    },
    
    visualMap: {
      min: 0, max: maxOverlap || 1, calculable: true, 
      orient: 'vertical', 
      right: '6%', 
      
      top: '10%', 
      
      itemWidth: 10, 
      itemHeight: 230, 
      
      text: ['', ''],
      textGap: 0,
      handleIcon: 'circle', 
      handleSize: '160%', 
      handleStyle: { color: '#3b82f6', borderColor: '#ffffff', borderWidth: 2, shadowBlur: 5, shadowColor: 'rgba(0,0,0,0.1)' },
      inRange: { color: ['#f8fafc', '#93c5fd', '#3b82f6', '#1e40af'] }
    },
    series: [{
      name: '同框时长', type: 'heatmap', data: matrixData,
      label: { show: true, formatter: p => p.data[2] === '-' ? '' : p.data[2] },
      itemStyle: { borderColor: '#fff', borderWidth: 2, borderRadius: 4 },
      emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0, 0, 0, 0.2)' } },
      animationDuration: 800,
    }]
  }
  myChart.setOption(option, true)
}

const prevPage = () => { if (currentPage.value > 0) { currentPage.value--; scheduleRender('gantt') } }
const nextPage = () => { if (currentPage.value < totalPages.value - 1) { currentPage.value++; scheduleRender('gantt') } }

onMounted(() => {
  resizeHandler = () => resizeAllCharts()
  window.addEventListener('resize', resizeHandler)
  refreshLayout()
})

onUnmounted(() => {
  if (pollInterval) clearInterval(pollInterval)
  if (renderFrame) cancelAnimationFrame(renderFrame)
  if (resizeTimer) clearTimeout(resizeTimer)
  if (layoutRefreshTimer) clearTimeout(layoutRefreshTimer)
  if (resizeHandler) window.removeEventListener('resize', resizeHandler)
  Object.values(chartInstances).forEach((instance) => instance?.dispose())
  releaseLocalUrl()
})
</script>

<style scoped>
.main-content { flex: 1; min-height: 0; display: flex; flex-direction: column; overflow: hidden; }
.content-area { flex: 1; min-height: 0; padding: 20px 40px; overflow-y: auto; display: flex; flex-direction: column; align-items: center; position: relative;}

.welcome-box { text-align: center; background: #ffffff; padding: 60px; border-radius: 12px; box-shadow: var(--shadow-light); border: 1px solid var(--border-color); width: 100%; max-width: 600px; margin-top: 10vh; transition: all 0.3s; cursor: pointer; }
.welcome-box:hover { border-color: var(--accent-primary); background-color: #f8fafc; }
.welcome-box h2 { color: var(--text-primary); margin-top: 20px; font-weight: 600; pointer-events: none;}
.welcome-box p { color: var(--text-secondary); margin-bottom: 30px; pointer-events: none;}
.upload-btn { width: 200px; border-radius: 8px; font-weight: bold; }

.progress-box { width: 100%; max-width: 600px; background: #ffffff; padding: 40px; border-radius: 12px; box-shadow: var(--shadow-light); border: 1px solid var(--border-color); text-align: center; margin-top: 10vh; }
.progress-box h2 { color: var(--text-primary); margin-bottom: 30px; font-weight: 600; }
.progress-tip { color: var(--text-secondary); font-size: 13px; margin-top: 20px; }

.result-workspace { width: 100%; max-width: 1400px; display: flex; flex-direction: column; gap: 20px; padding: 10px 0; }
.stats-panel { width: 100%; }
.history-video-alert { width: 100%; }

.stat-card { background-color: #ffffff; border-radius: 12px; padding: 20px; box-shadow: var(--shadow-light); border: 1px solid var(--border-color); transition: all 0.2s ease; }
.stat-card:hover { box-shadow: var(--shadow-medium); transform: translateY(-2px); }
.stat-item { text-align: center; }
.stat-label { font-size: 14px; color: var(--text-secondary); margin-bottom: 8px; }
.stat-value { font-size: 24px; font-weight: 600; color: var(--text-primary); }

.charts-container { width: 100%; display: flex; flex-direction: column; gap: 24px; }
.chart-row { display: flex; gap: 24px; width: 100%; }
.chart-box { flex: 1; background: #ffffff; border-radius: 12px; padding: 24px; box-shadow: var(--shadow-light); border: 1px solid var(--border-color); display: flex; flex-direction: column; }
.chart-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;}
.chart-title { margin: 0; color: var(--text-primary); font-size: 16px; display: flex; align-items: center; gap: 8px; }

/* 馃殌 鍔ㄦ€佽繃婊ゆ粦鍧楁牱寮?*/
.filter-control { display: flex; align-items: center; background: #f8fafc; padding: 4px 16px; border-radius: 8px; border: 1px solid var(--border-color);}
.filter-label { font-size: 13px; color: var(--text-secondary); font-weight: 500;}

.video-container { background: #000; border-radius: 12px; overflow: hidden; box-shadow: var(--shadow-medium); width: 100%; aspect-ratio: 16/9; max-height: 500px; display: flex; justify-content: center; align-items: center; margin-bottom: 10px;}
.video-container.disabled { background: linear-gradient(135deg, rgba(18, 24, 38, 0.96), rgba(35, 44, 66, 0.92)); }
.anime-video { width: 100%; height: 100%; outline: none; }
.video-empty-state { display: flex; align-items: center; justify-content: center; width: 100%; height: 100%; color: rgba(255, 255, 255, 0.78); font-size: 15px; padding: 24px; text-align: center; }

.pagination { display: flex; justify-content: center; align-items: center; gap: 12px; margin-top: 16px; font-size: 14px; color: var(--text-secondary); }
.pagination button { padding: 6px 14px; background: #f0f0f0; border: none; border-radius: 6px; cursor: pointer; transition: all 0.2s ease; color: var(--text-primary);}
.pagination button:hover:not(:disabled) { background: #e0e0e0; }

.floating-progress { position: fixed; bottom: 40px; right: 40px; width: 320px; background: #ffffff; padding: 20px; border-radius: 12px; box-shadow: var(--shadow-medium); border: 1px solid var(--border-color); z-index: 1000; animation: slideIn 0.4s cubic-bezier(0.16, 1, 0.3, 1); }
.fp-header { display: flex; justify-content: space-between; margin-bottom: 15px; font-size: 14px; font-weight: 600; color: var(--accent-secondary); }
@keyframes slideIn { from { transform: translateY(50px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
</style>

