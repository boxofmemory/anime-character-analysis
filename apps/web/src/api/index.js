import axios from 'axios'

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  import.meta.env.VITE_API_URL ||
  'http://localhost:8000'

export const toApiUrl = (path) => {
  if (!path) {
    return ''
  }
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return path
  }
  return `${API_BASE_URL}${path}`
}

export const apiUploadVideo = (file, modelSelection = null, dataExportEnabled = false) => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('data_export_enabled', dataExportEnabled ? 'true' : 'false')
  if (modelSelection?.detection_model_path) {
    formData.append('detection_model_path', modelSelection.detection_model_path)
  }
  if (modelSelection?.recognition_model_path) {
    formData.append('recognition_model_path', modelSelection.recognition_model_path)
  }
  if (modelSelection?.tracker_config_path) {
    formData.append('tracker_config_path', modelSelection.tracker_config_path)
  }
  return axios.post(`${API_BASE_URL}/api/upload`, formData)
}

export const apiGetVideoStatus = (videoId) => {
  return axios.get(`${API_BASE_URL}/api/videos/${videoId}/status`)
}

export const apiGetTimeline = (videoId) => {
  return axios.get(`${API_BASE_URL}/api/videos/${videoId}/timeline`)
}

export const apiGetHistory = () => {
  return axios.get(`${API_BASE_URL}/api/history`)
}

export const apiGetAnalysisModelOptions = () => {
  return axios.get(`${API_BASE_URL}/api/analysis/model-options`)
}

export const apiGetHealthReady = () => {
  return axios.get(`${API_BASE_URL}/api/health/ready`)
}

export const apiRecognizeImage = (file, modelSelection = null) => {
  const formData = new FormData()
  formData.append('file', file)
  if (modelSelection?.detection_model_path) {
    formData.append('detection_model_path', modelSelection.detection_model_path)
  }
  if (modelSelection?.recognition_model_path) {
    formData.append('recognition_model_path', modelSelection.recognition_model_path)
  }
  return axios.post(`${API_BASE_URL}/api/image-recognition`, formData)
}

export const apiRenameStranger = (oldName, newName, recognitionModelPath = null) => {
  return axios.post(`${API_BASE_URL}/api/stranger/rename`, {
    old_name: oldName,
    new_name: newName,
    recognition_model_path: recognitionModelPath,
  })
}

export const apiRenameVideo = (videoId, newName) => {
  return axios.put(`${API_BASE_URL}/api/videos/${videoId}/rename`, {
    filename: newName,
  })
}

export const apiDeleteVideo = (videoId) => {
  return axios.delete(`${API_BASE_URL}/api/videos/${videoId}`)
}

export const apiGetDashboardSummary = () => {
  return axios.get(`${API_BASE_URL}/api/dashboard/summary`)
}

export const apiGetModelShowcase = (modelSelection = null) => {
  return axios.get(`${API_BASE_URL}/api/model-showcase`, {
    params: {
      detection_model_path: modelSelection?.detection_model_path || undefined,
      recognition_model_path: modelSelection?.recognition_model_path || undefined,
      tracker_config_path: modelSelection?.tracker_config_path || undefined,
    },
  })
}

export const apiGetTrainingOptions = () => {
  return axios.get(`${API_BASE_URL}/api/training/options`)
}

export const apiCreateTrainingJob = (payload) => {
  return axios.post(`${API_BASE_URL}/api/training/jobs`, payload)
}

export const apiGetTrainingJobs = () => {
  return axios.get(`${API_BASE_URL}/api/training/jobs`)
}

export const apiGetTrainingJob = (jobId) => {
  return axios.get(`${API_BASE_URL}/api/training/jobs/${encodeURIComponent(jobId)}`)
}

export const apiGetTrainingLog = (jobId) => {
  return axios.get(`${API_BASE_URL}/api/training/jobs/${encodeURIComponent(jobId)}/log`)
}

export const apiPromoteTrainingJob = (jobId) => {
  return axios.post(`${API_BASE_URL}/api/training/jobs/${encodeURIComponent(jobId)}/promote`)
}

export const apiCancelTrainingJob = (jobId) => {
  return axios.post(`${API_BASE_URL}/api/training/jobs/${encodeURIComponent(jobId)}/cancel`)
}

export const apiGetLibraryOptions = (recognitionModelKey = null) => {
  return axios.get(`${API_BASE_URL}/api/library/options`, {
    params: recognitionModelKey ? { recognition_model_key: recognitionModelKey } : {},
  })
}

export const apiGetLibraryCharacters = (recognitionModelKey = null) => {
  return axios.get(`${API_BASE_URL}/api/library/characters`, {
    params: recognitionModelKey ? { recognition_model_key: recognitionModelKey } : {},
  })
}

export const apiGetLibraryCharacterDetail = (name, recognitionModelKey = null) => {
  return axios.get(`${API_BASE_URL}/api/library/characters/${encodeURIComponent(name)}`, {
    params: recognitionModelKey ? { recognition_model_key: recognitionModelKey } : {},
  })
}

export const apiImportLibraryCharacter = (formData) => {
  return axios.post(`${API_BASE_URL}/api/library/characters/import`, formData)
}

export const apiImportLibraryCharacterFromWeb = (formData) => {
  return axios.post(`${API_BASE_URL}/api/library/characters/import-from-web`, formData)
}

export const apiRebuildLibraryCharacter = (name, recognitionModelKey = null) => {
  return axios.post(
    `${API_BASE_URL}/api/library/characters/${encodeURIComponent(name)}/rebuild`,
    null,
    {
      params: recognitionModelKey ? { recognition_model_key: recognitionModelKey } : {},
    },
  )
}

export const apiDeleteLibraryCharacter = (name, recognitionModelKey = null) => {
  return axios.delete(`${API_BASE_URL}/api/library/characters/${encodeURIComponent(name)}`, {
    params: recognitionModelKey ? { recognition_model_key: recognitionModelKey } : {},
  })
}
