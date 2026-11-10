import { computed, onUnmounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { getTimeline, getVideoStatus, getAnnotatedVideoUrl, uploadVideo } from '../services/videoService'

export function useVideoAnalysis() {
  const status = ref('idle')
  const progress = ref(0)
  const processingTime = ref(0)
  const isViewingHistory = ref(false)
  const isBackgroundTaskRunning = ref(false)
  const timelineData = ref({})
  const annotatedVideoUrl = ref('')
  const localVideoUrl = ref('')
  const activeVideoId = ref(null)

  let pollTimer = null
  let startTime = 0

  const filteredTimelineData = computed(() => timelineData.value || {})

  function revokeLocalUrl() {
    if (localVideoUrl.value) {
      URL.revokeObjectURL(localVideoUrl.value)
      localVideoUrl.value = ''
    }
  }

  function resetForNewTask() {
    if (pollTimer) {
      clearInterval(pollTimer)
      pollTimer = null
    }
    status.value = 'idle'
    progress.value = 0
    processingTime.value = 0
    timelineData.value = {}
    annotatedVideoUrl.value = ''
    activeVideoId.value = null
    isViewingHistory.value = false
    isBackgroundTaskRunning.value = false
    revokeLocalUrl()
  }

  async function loadHistoryData(videoId) {
    status.value = 'loading_history'
    isViewingHistory.value = true
    activeVideoId.value = videoId
    annotatedVideoUrl.value = getAnnotatedVideoUrl(videoId)
    const data = await getTimeline(videoId)
    timelineData.value = data.timeline
    status.value = 'completed'
  }

  async function uploadAndAnalyze(file) {
    revokeLocalUrl()
    localVideoUrl.value = URL.createObjectURL(file)
    status.value = 'uploading'
    progress.value = 0
    processingTime.value = 0
    timelineData.value = {}
    isViewingHistory.value = false
    isBackgroundTaskRunning.value = true

    const result = await uploadVideo(file)
    activeVideoId.value = result.video_id
    annotatedVideoUrl.value = getAnnotatedVideoUrl(result.video_id)
    status.value = 'processing'
    startPolling()
  }

  function startPolling() {
    if (pollTimer) {
      clearInterval(pollTimer)
    }
    startTime = Date.now()
    pollTimer = window.setInterval(async () => {
      try {
        const data = await getVideoStatus(activeVideoId.value)
        progress.value = data.progress || 0
        processingTime.value = Math.floor((Date.now() - startTime) / 1000)

        if (data.status === 'completed') {
          clearInterval(pollTimer)
          pollTimer = null
          isBackgroundTaskRunning.value = false
          const timeline = await getTimeline(activeVideoId.value)
          timelineData.value = timeline.timeline
          status.value = 'completed'
        } else if (data.status === 'failed') {
          clearInterval(pollTimer)
          pollTimer = null
          isBackgroundTaskRunning.value = false
          status.value = 'idle'
          ElMessage.error('AI 处理失败，请检查后端日志')
        }
      } catch (error) {
        console.error('Polling failed', error)
      }
    }, 2000)
  }

  onUnmounted(() => {
    if (pollTimer) {
      clearInterval(pollTimer)
    }
    revokeLocalUrl()
  })

  return {
    activeVideoId,
    annotatedVideoUrl,
    filteredTimelineData,
    isBackgroundTaskRunning,
    isViewingHistory,
    localVideoUrl,
    processingTime,
    progress,
    status,
    timelineData,
    loadHistoryData,
    resetForNewTask,
    uploadAndAnalyze,
  }
}
