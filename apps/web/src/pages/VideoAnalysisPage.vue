<template>
  <div class="workspace">
    <section v-if="analysis.status === 'idle'" class="hero-card" @dragover.prevent @drop="handleDrop">
      <p class="eyebrow">Video Pipeline</p>
      <h2>上传动漫视频，生成角色时间轴与共现分析</h2>
      <p class="hero-copy">自动完成角色检测、身份识别、轨迹跟踪、时长统计与多维图表展示。</p>
      <input ref="fileInput" type="file" accept="video/*" hidden @change="handleFileChange" />
      <div class="hero-actions">
        <el-button type="primary" size="large" @click="fileInput?.click()">选择视频</el-button>
        <span>支持 mp4、webm、ts 等常见视频格式</span>
      </div>
    </section>

    <section v-else-if="analysis.status === 'uploading' || analysis.status === 'processing'" class="status-card">
      <h2>{{ analysis.status === 'uploading' ? '正在上传视频' : 'AI 正在分析视频' }}</h2>
      <el-progress :percentage="analysis.progress" :stroke-width="20" striped striped-flow />
      <p>已耗时 {{ analysis.processingTime }} 秒</p>
    </section>

    <section v-else-if="analysis.status === 'loading_history'" class="status-card">
      <h2>正在加载历史结果</h2>
    </section>

    <section v-else-if="analysis.status === 'completed'" class="result-grid">
      <div class="summary-grid">
        <article class="summary-card">
          <span>角色数</span>
          <strong>{{ characterDurations.length }}</strong>
        </article>
        <article class="summary-card">
          <span>已知角色</span>
          <strong>{{ knownCount }}</strong>
        </article>
        <article class="summary-card">
          <span>陌生人</span>
          <strong>{{ strangerCount }}</strong>
        </article>
        <article class="summary-card">
          <span>当前视频</span>
          <strong>{{ analysis.activeVideoId || '-' }}</strong>
        </article>
      </div>

      <div class="video-card">
        <video
          ref="videoRef"
          :src="analysis.annotatedVideoUrl || analysis.localVideoUrl"
          controls
          class="video-player"
        />
      </div>

      <div class="chart-card chart-wide">
        <div class="card-head">
          <h3>角色出场时间轴</h3>
          <span>点击色块可跳转到视频对应时间</span>
        </div>
        <div id="timeline-chart" class="chart"></div>
      </div>

      <div class="chart-card">
        <div class="card-head">
          <h3>出场时长占比</h3>
        </div>
        <div id="pie-chart" class="chart"></div>
      </div>

      <div class="chart-card">
        <div class="card-head">
          <h3>角色时长排行</h3>
        </div>
        <div id="bar-chart" class="chart"></div>
      </div>

      <div class="chart-card chart-wide">
        <div class="card-head">
          <h3>角色共现热力图</h3>
        </div>
        <div id="heatmap-chart" class="chart heatmap-chart"></div>
      </div>

      <div class="chart-card chart-wide">
        <div class="card-head">
          <h3>陌生人转正</h3>
          <span>点击下方按钮可将陌生人合并或转正为已知角色</span>
        </div>
        <div class="stranger-actions">
          <el-button
            v-for="item in strangerCharacters"
            :key="item.name"
            @click="renameCharacter(item.name)"
          >
            {{ translateName(item.name) }}
          </el-button>
          <span v-if="!strangerCharacters.length">当前结果中没有陌生人角色。</span>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts'
import { ElMessage, ElMessageBox } from 'element-plus'
import { renameStranger } from '../services/characterService'
import { getCharacterDurations, translateName } from '../utils/character'

const props = defineProps({
  analysis: {
    type: Object,
    required: true,
  },
})

const emit = defineEmits(['history-updated'])
const fileInput = ref(null)
const videoRef = ref(null)
const chartInstances = []

const characterDurations = computed(() => getCharacterDurations(props.analysis.filteredTimelineData))
const knownCount = computed(() => characterDurations.value.filter((item) => item.type === 'known').length)
const strangerCount = computed(() => characterDurations.value.filter((item) => item.type === 'stranger').length)
const strangerCharacters = computed(() => characterDurations.value.filter((item) => item.name.startsWith('Stranger_')))

function disposeCharts() {
  while (chartInstances.length) {
    const chart = chartInstances.pop()
    chart.dispose()
  }
}

function initChart(id) {
  const dom = document.getElementById(id)
  if (!dom) return null
  const chart = echarts.init(dom)
  chartInstances.push(chart)
  return chart
}

function renderTimelineChart() {
  const chart = initChart('timeline-chart')
  if (!chart) return
  const labels = characterDurations.value.map((item) => item.name)
  const seriesData = []
  labels.forEach((name, index) => {
    props.analysis.filteredTimelineData[name].segments.forEach((segment) => {
      seriesData.push({
        name,
        value: [index, segment.start, segment.end, segment.end - segment.start],
        itemStyle: {
          color: name.startsWith('Stranger_') ? '#f59e0b' : '#0f766e',
        },
      })
    })
  })
  chart.setOption({
    tooltip: {
      formatter: (params) => `${translateName(params.data.name)}<br/>${params.value[1].toFixed(2)}s - ${params.value[2].toFixed(2)}s`,
    },
    grid: { left: '3%', right: '3%', top: 20, bottom: 20, containLabel: true },
    xAxis: { type: 'value', axisLabel: { formatter: '{value}s' } },
    yAxis: {
      type: 'category',
      inverse: true,
      data: labels.map((name) => translateName(name)),
    },
    series: [
      {
        type: 'custom',
        renderItem(params, api) {
          const categoryIndex = api.value(0)
          const start = api.coord([api.value(1), categoryIndex])
          const end = api.coord([api.value(2), categoryIndex])
          const height = api.size([0, 1])[1] * 0.5
          return {
            type: 'rect',
            shape: {
              x: start[0],
              y: start[1] - height / 2,
              width: end[0] - start[0],
              height,
            },
            style: api.style(),
          }
        },
        encode: { x: [1, 2], y: 0 },
        data: seriesData,
      },
    ],
  })
  chart.on('click', (params) => {
    if (videoRef.value) {
      videoRef.value.currentTime = params.value[1]
      videoRef.value.play()
    }
  })
}

function renderPieChart() {
  const chart = initChart('pie-chart')
  if (!chart) return
  chart.setOption({
    tooltip: { trigger: 'item' },
    series: [
      {
        type: 'pie',
        radius: ['40%', '70%'],
        data: characterDurations.value.map((item) => ({
          name: translateName(item.name),
          value: item.total,
          itemStyle: { color: item.type === 'known' ? '#0f766e' : '#f59e0b' },
        })),
      },
    ],
  })
}

function renderBarChart() {
  const chart = initChart('bar-chart')
  if (!chart) return
  const top = characterDurations.value.slice(0, 10)
  chart.setOption({
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'category',
      data: top.map((item) => translateName(item.name)),
      axisLabel: { interval: 0, rotate: 25 },
    },
    yAxis: { type: 'value' },
    grid: { left: '3%', right: '3%', bottom: '15%', containLabel: true },
    series: [
      {
        type: 'bar',
        data: top.map((item) => ({
          value: item.total,
          itemStyle: { color: item.type === 'known' ? '#0f766e' : '#f59e0b' },
        })),
      },
    ],
  })
}

function renderHeatmapChart() {
  const chart = initChart('heatmap-chart')
  if (!chart) return
  const names = characterDurations.value.slice(0, 10).map((item) => item.name)
  const matrix = []
  let max = 1
  names.forEach((nameA, y) => {
    names.forEach((nameB, x) => {
      if (nameA === nameB) {
        matrix.push([x, y, 0])
        return
      }
      let overlap = 0
      props.analysis.filteredTimelineData[nameA].segments.forEach((segA) => {
        props.analysis.filteredTimelineData[nameB].segments.forEach((segB) => {
          const start = Math.max(segA.start, segB.start)
          const end = Math.min(segA.end, segB.end)
          if (start < end) overlap += end - start
        })
      })
      max = Math.max(max, overlap)
      matrix.push([x, y, Number(overlap.toFixed(2))])
    })
  })
  chart.setOption({
    tooltip: {
      formatter: (params) => `${translateName(names[params.data[1]])} / ${translateName(names[params.data[0]])}<br/>共现 ${params.data[2]} 秒`,
    },
    grid: { left: '10%', right: '15%', top: 20, bottom: 40 },
    xAxis: { type: 'category', data: names.map(translateName), axisLabel: { rotate: 30 } },
    yAxis: { type: 'category', data: names.map(translateName) },
    visualMap: {
      min: 0,
      max,
      calculable: true,
      orient: 'vertical',
      right: 0,
      top: 'middle',
      inRange: { color: ['#f1f5f9', '#93c5fd', '#1d4ed8'] },
    },
    series: [
      {
        type: 'heatmap',
        data: matrix,
        label: { show: true },
      },
    ],
  })
}

function renderCharts() {
  disposeCharts()
  nextTick(() => {
    renderTimelineChart()
    renderPieChart()
    renderBarChart()
    renderHeatmapChart()
  })
}

async function renameCharacter(oldName) {
  try {
    const { value } = await ElMessageBox.prompt('请输入正式角色英文名，例如 Hoshino_Ai', '陌生人转正')
    if (!value) return
    await renameStranger(oldName, value)
    ElMessage.success('角色库已更新')
    if (props.analysis.activeVideoId) {
      await props.analysis.loadHistoryData(props.analysis.activeVideoId)
    }
    emit('history-updated')
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error(error.response?.data?.detail || error.message || '转正失败')
    }
  }
}

function handleFileChange(event) {
  const file = event.target.files?.[0]
  if (file) {
    props.analysis.uploadAndAnalyze(file).then(() => emit('history-updated'))
  }
}

function handleDrop(event) {
  const file = event.dataTransfer?.files?.[0]
  if (file) {
    props.analysis.uploadAndAnalyze(file).then(() => emit('history-updated'))
  }
}

watch(
  () => props.analysis.status,
  (value) => {
    if (value === 'completed') {
      renderCharts()
    } else {
      disposeCharts()
    }
  }
)

onMounted(() => {
  if (props.analysis.status === 'completed') {
    renderCharts()
  }
})

onBeforeUnmount(disposeCharts)
</script>
