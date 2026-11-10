<template>
  <div class="chart-card">
    <div class="chart-head">
      <strong>{{ chart.title }}</strong>
      <span v-if="chart.summary">{{ chart.summary }}</span>
    </div>
    <div ref="chartRef" class="chart-canvas"></div>
  </div>
</template>

<script setup>
import * as echarts from 'echarts'
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps({
  chart: {
    type: Object,
    required: true,
  },
})

const chartRef = ref(null)
let chartInstance = null
let resizeObserver = null
let intersectionObserver = null
let renderFrame = 0

const getLabelInterval = (labelCount) => {
  if (labelCount <= 10) return 0
  return Math.ceil(labelCount / 8) - 1
}

const buildLineOption = (chart) => {
  const labels = chart.x_labels || []
  return {
    color: ['#0f766e', '#2563eb', '#f59e0b', '#ef4444', '#8b5cf6'],
    tooltip: { trigger: 'axis' },
    legend: {
      top: 2,
      type: 'scroll',
      textStyle: { color: '#6e6e80', fontSize: 12 },
    },
    grid: {
      left: 44,
      right: 18,
      top: 64,
      bottom: 34,
      containLabel: true,
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: labels,
      axisLabel: {
        color: '#6e6e80',
        fontSize: 11,
        interval: getLabelInterval(labels.length),
      },
      axisLine: { lineStyle: { color: '#d4d8e3' } },
    },
    yAxis: {
      type: 'value',
      name: chart.y_axis_label || '',
      axisLabel: { color: '#6e6e80', fontSize: 11 },
      splitLine: { lineStyle: { color: '#eef2f7' } },
    },
    series: (chart.series || []).map((item) => ({
      name: item.name,
      type: 'line',
      smooth: true,
      symbol: 'circle',
      symbolSize: 6,
      connectNulls: true,
      emphasis: { focus: 'series' },
      areaStyle: { opacity: 0.06 },
      data: item.values || [],
    })),
  }
}

const buildBarOption = (chart) => {
  const labels = chart.x_labels || []
  return {
    color: ['#0f766e', '#2563eb', '#f59e0b', '#ef4444', '#8b5cf6'],
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: {
      top: 2,
      type: 'scroll',
      textStyle: { color: '#6e6e80', fontSize: 12 },
    },
    grid: {
      left: 118,
      right: 18,
      top: 58,
      bottom: 28,
      containLabel: false,
    },
    xAxis: {
      type: 'value',
      name: chart.y_axis_label || '',
      axisLabel: { color: '#6e6e80', fontSize: 11 },
      splitLine: { lineStyle: { color: '#eef2f7' } },
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: labels,
      axisLabel: {
        color: '#6e6e80',
        fontSize: 11,
        width: 108,
        overflow: 'truncate',
      },
      axisLine: { lineStyle: { color: '#d4d8e3' } },
    },
    series: (chart.series || []).map((item) => ({
      name: item.name,
      type: 'bar',
      barMaxWidth: 18,
      emphasis: { focus: 'series' },
      data: item.values || [],
    })),
  }
}

const renderChart = () => {
  if (!chartRef.value || !props.chart) {
    return
  }
  if (chartRef.value.clientWidth < 40 || chartRef.value.clientHeight < 40) {
    return
  }
  if (!chartInstance) {
    chartInstance = echarts.init(chartRef.value)
  }

  const isBar = props.chart.chart_type === 'bar'
  chartInstance.setOption(isBar ? buildBarOption(props.chart) : buildLineOption(props.chart), true)
  chartInstance.resize()
}

const scheduleRender = async () => {
  await nextTick()
  if (renderFrame) {
    cancelAnimationFrame(renderFrame)
  }
  renderFrame = requestAnimationFrame(() => {
    renderFrame = 0
    renderChart()
  })
}

const handleResize = () => {
  scheduleRender()
}

onMounted(() => {
  resizeObserver = new ResizeObserver(() => {
    scheduleRender()
  })
  if (chartRef.value) {
    resizeObserver.observe(chartRef.value)
  }

  intersectionObserver = new IntersectionObserver((entries) => {
    if (entries.some((entry) => entry.isIntersecting)) {
      scheduleRender()
    }
  })
  if (chartRef.value) {
    intersectionObserver.observe(chartRef.value)
  }

  scheduleRender()
  window.addEventListener('resize', handleResize)
})

watch(
  () => props.chart,
  () => {
    scheduleRender()
  },
  { deep: true },
)

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize)
  resizeObserver?.disconnect()
  intersectionObserver?.disconnect()
  if (renderFrame) {
    cancelAnimationFrame(renderFrame)
  }
  chartInstance?.dispose()
  chartInstance = null
})
</script>

<style scoped>
.chart-card {
  padding: 16px;
  border: 1px solid var(--border-color);
  border-radius: 14px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(248, 250, 252, 0.92));
  min-width: 0;
  overflow: hidden;
}

.chart-head {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 10px;
}

.chart-head strong {
  font-size: 15px;
  color: var(--text-primary);
}

.chart-head span {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.6;
}

.chart-canvas {
  width: 100%;
  height: 280px;
  min-width: 0;
}

@media (max-width: 720px) {
  .chart-canvas {
    height: 320px;
  }
}
</style>
