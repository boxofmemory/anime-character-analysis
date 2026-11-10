<template>
  <div class="app-wrapper">
    <Sidebar
      :current-page="currentPage"
      :analysis-model-selection="analysisModelSelection"
      @navigate="handleNavigate"
      @new-task="triggerNewTask"
      @select-history="loadHistoryRecord"
      @history-deleted="handleHistoryDeleted"
      @model-selection-change="handleModelSelectionChange"
    />

    <main class="main-content">
      <AppTopBar
        v-if="currentPage === 'video'"
        :current-mode="currentMode"
        @mode-change="onModeChange"
      />

      <section v-show="currentPage === 'video'" class="page-shell">
        <VideoView
          v-show="currentMode === 'video'"
          ref="videoViewRef"
          :analysis-model-selection="analysisModelSelection"
          :data-export-enabled="dataExportEnabled"
        />

        <ImageRecognitionView
          v-if="currentMode === 'image'"
          :analysis-model-selection="analysisModelSelection"
        />

        <div v-else-if="currentMode !== 'video'" class="coming-soon">
          <div class="coming-soon-card">
            <h2>链接解析模块暂未启用</h2>
            <p>当前系统重点提供视频时序分析与单图角色识别能力。</p>
          </div>
        </div>
      </section>

      <DashboardPage
        v-if="currentPage === 'dashboard'"
        @open-history="loadHistoryRecord"
        @open-settings-section="openSettingsSection"
      />

      <SettingsPage
        v-if="currentPage === 'settings'"
        :current-theme="currentTheme"
        :target-section="settingsTargetSection"
        :analysis-model-selection="analysisModelSelection"
        :data-export-enabled="dataExportEnabled"
        @theme-change="handleThemeChange"
        @analysis-model-selection-change="handleModelSelectionChange"
        @data-export-change="handleDataExportChange"
      />
    </main>
  </div>
</template>

<script setup>
import { nextTick, onMounted, ref, watch } from 'vue'
import Sidebar from './components/layout/Sidebar.vue'
import AppTopBar from './components/layout/AppTopBar.vue'
import DashboardPage from './pages/DashboardPage.vue'
import SettingsPage from './pages/SettingsPage.vue'
import ImageRecognitionView from './views/ImageRecognitionView.vue'
import VideoView from './views/VideoView.vue'

const THEME_STORAGE_KEY = 'act-theme'
const DATA_EXPORT_STORAGE_KEY = 'act-data-export-enabled'

const themePresets = {
  moonlightBlue: {
    '--bg-primary': '#f4f5f9',
    '--bg-secondary': '#ffffff',
    '--surface-primary': '#ffffff',
    '--surface-secondary': '#fafafa',
    '--text-primary': '#202123',
    '--text-secondary': '#6e6e80',
    '--accent-primary': '#10a37f',
    '--accent-secondary': '#10a37f',
    '--accent-danger': '#ef4444',
    '--border-color': 'rgba(0, 0, 0, 0.08)',
    '--shadow-light': '0 2px 6px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02)',
    '--shadow-medium': '0 6px 12px rgba(0, 0, 0, 0.06), 0 2px 4px rgba(0, 0, 0, 0.04)',
    '--hero-gradient': '#ffffff',
  },
  mistGreen: {
    '--bg-primary': '#f4f5f9',
    '--bg-secondary': '#ffffff',
    '--surface-primary': '#ffffff',
    '--surface-secondary': '#fafafa',
    '--text-primary': '#202123',
    '--text-secondary': '#6e6e80',
    '--accent-primary': '#0f172a',
    '--accent-secondary': '#334155',
    '--accent-danger': '#ef4444',
    '--border-color': 'rgba(0, 0, 0, 0.08)',
    '--shadow-light': '0 2px 6px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02)',
    '--shadow-medium': '0 6px 12px rgba(0, 0, 0, 0.06), 0 2px 4px rgba(0, 0, 0, 0.04)',
    '--hero-gradient': '#ffffff',
  },
  warmGold: {
    '--bg-primary': '#f5f5f5',
    '--bg-secondary': '#ffffff',
    '--surface-primary': '#ffffff',
    '--surface-secondary': '#fafafa',
    '--text-primary': '#1a1a1a',
    '--text-secondary': '#666666',
    '--accent-primary': '#3b82f6',
    '--accent-secondary': '#2563eb',
    '--accent-danger': '#ef4444',
    '--border-color': 'rgba(0, 0, 0, 0.08)',
    '--shadow-light': '0 2px 6px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02)',
    '--shadow-medium': '0 6px 12px rgba(0, 0, 0, 0.06), 0 2px 4px rgba(0, 0, 0, 0.04)',
    '--hero-gradient': '#ffffff',
  },
}

const currentPage = ref('video')
const currentMode = ref('video')
const currentTheme = ref('moonlightBlue')
const videoViewRef = ref(null)
const currentHistoryId = ref(null)
const settingsTargetSection = ref('library')
const dataExportEnabled = ref(false)
const analysisModelSelection = ref({
  detection_model_path: '',
  recognition_model_path: '',
  tracker_config_path: '',
})
let videoRefreshTimer = null

const applyTheme = (themeName) => {
  const preset = themePresets[themeName] || themePresets.moonlightBlue
  currentTheme.value = themePresets[themeName] ? themeName : 'moonlightBlue'
  Object.entries(preset).forEach(([key, value]) => {
    document.documentElement.style.setProperty(key, value)
  })
  localStorage.setItem(THEME_STORAGE_KEY, currentTheme.value)
}

const handleThemeChange = (themeName) => {
  applyTheme(themeName)
}

const handleNavigate = (page) => {
  currentPage.value = page
}

const handleModelSelectionChange = (selection) => {
  analysisModelSelection.value = {
    detection_model_path: selection?.detection_model_path || '',
    recognition_model_path: selection?.recognition_model_path || '',
    tracker_config_path: selection?.tracker_config_path || '',
  }
}

const handleDataExportChange = (enabled) => {
  dataExportEnabled.value = Boolean(enabled)
  localStorage.setItem(DATA_EXPORT_STORAGE_KEY, dataExportEnabled.value ? 'true' : 'false')
}

const refreshVideoLayout = () => {
  nextTick(() => {
    videoViewRef.value?.refreshLayout?.()
    if (videoRefreshTimer) {
      clearTimeout(videoRefreshTimer)
    }
    videoRefreshTimer = setTimeout(() => {
      videoViewRef.value?.refreshLayout?.()
      videoRefreshTimer = null
    }, 160)
  })
}

const openSettingsSection = (section = 'library') => {
  settingsTargetSection.value = section
  currentPage.value = 'settings'
}

const onModeChange = (newMode) => {
  currentMode.value = newMode
}

const triggerNewTask = () => {
  currentHistoryId.value = null
  currentPage.value = 'video'
  currentMode.value = 'video'
  videoViewRef.value?.resetForNewTask?.()
  refreshVideoLayout()
}

const loadHistoryRecord = (historyItem) => {
  currentHistoryId.value = historyItem?.id ?? null
  currentPage.value = 'video'
  currentMode.value = 'video'
  videoViewRef.value?.loadHistoryData?.(historyItem)
  refreshVideoLayout()
}

const handleHistoryDeleted = (deletedId) => {
  if (currentHistoryId.value === deletedId) {
    triggerNewTask()
  }
}

onMounted(() => {
  applyTheme(localStorage.getItem(THEME_STORAGE_KEY) || 'moonlightBlue')
  dataExportEnabled.value = localStorage.getItem(DATA_EXPORT_STORAGE_KEY) === 'true'
})

watch(currentPage, (page) => {
  if (page === 'video') {
    refreshVideoLayout()
  }
})

watch(currentMode, (mode) => {
  if (mode === 'video' && currentPage.value === 'video') {
    refreshVideoLayout()
  }
})
</script>

<style>
:root {
  --bg-primary: #f4f5f9;
  --bg-secondary: #ffffff;
  --surface-primary: #ffffff;
  --surface-secondary: #fafafa;
  --text-primary: #202123;
  --text-secondary: #6e6e80;
  --accent-primary: #10a37f;
  --accent-secondary: #10a37f;
  --accent-danger: #ef4444;
  --border-color: rgba(0, 0, 0, 0.08);
  --shadow-light: 0 2px 6px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02);
  --shadow-medium: 0 6px 12px rgba(0, 0, 0, 0.06), 0 2px 4px rgba(0, 0, 0, 0.04);
  --hero-gradient: #ffffff;
}

html,
body,
#app {
  margin: 0;
  min-height: 100%;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
}

* {
  box-sizing: border-box;
}

body {
  overflow: hidden;
}

.app-wrapper {
  display: flex;
  width: 100vw;
  height: 100vh;
  background: var(--bg-primary);
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
  background: var(--bg-primary);
}

.page-shell {
  flex: 1;
  min-height: 0;
  display: flex;
  overflow: hidden;
}

.coming-soon {
  height: calc(100vh - 60px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 32px;
}

.coming-soon-card {
  width: min(560px, 100%);
  padding: 40px;
  border-radius: 28px;
  background: var(--surface-primary);
  border: 1px solid var(--border-color);
  box-shadow: var(--shadow-medium);
  backdrop-filter: blur(14px);
  text-align: center;
}

.coming-soon-card h2 {
  margin: 0 0 12px;
  font-size: 28px;
}

.coming-soon-card p {
  margin: 0;
  color: var(--text-secondary);
  font-size: 15px;
}

.el-button {
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.el-button:hover {
  transform: translateY(-1px);
  box-shadow: var(--shadow-light);
}
</style>
