import { http } from './http'

export async function uploadVideo(file, dataExportEnabled = false) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('data_export_enabled', dataExportEnabled ? 'true' : 'false')
  const { data } = await http.post('/api/upload', formData)
  return data
}

export async function getVideoStatus(videoId) {
  const { data } = await http.get(`/api/videos/${videoId}/status`)
  return data
}

export async function getTimeline(videoId) {
  const { data } = await http.get(`/api/videos/${videoId}/timeline`)
  return data
}

export function getAnnotatedVideoUrl(videoId) {
  return `${http.defaults.baseURL}/api/videos/${videoId}/download`
}
