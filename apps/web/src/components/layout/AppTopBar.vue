<template>
  <header class="top-bar">
    <div class="headline">
      <h1>视频分析工作台</h1>
      <p>面向动漫视频的角色检测、身份识别、轨迹跟踪与时间轴分析。</p>
    </div>

    <el-radio-group v-model="activeMode" size="large" @change="handleModeChange">
      <el-radio-button value="video">视频时序分析</el-radio-button>
      <el-radio-button value="image">单图角色识别</el-radio-button>
      <el-radio-button value="url" disabled>链接解析</el-radio-button>
    </el-radio-group>
  </header>
</template>

<script setup>
import { ref, watch } from 'vue'

const props = defineProps({
  currentMode: {
    type: String,
    default: 'video',
  },
})

const activeMode = ref(props.currentMode)
const emit = defineEmits(['mode-change'])

const handleModeChange = (mode) => {
  emit('mode-change', mode)
}

watch(
  () => props.currentMode,
  (mode) => {
    activeMode.value = mode || 'video'
  }
)
</script>

<style scoped>
.top-bar {
  height: 78px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  padding: 0 28px;
  background: rgba(255, 255, 255, 0.72);
  border-bottom: 1px solid var(--border-color);
  backdrop-filter: blur(12px);
}

.headline h1 {
  margin: 0;
  font-size: 20px;
  line-height: 1.2;
}

.headline p {
  margin: 6px 0 0;
  color: var(--text-secondary);
  font-size: 13px;
}
</style>
