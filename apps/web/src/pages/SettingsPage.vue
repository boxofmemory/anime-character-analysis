<template>
  <section class="settings-page">
    <div class="settings-shell">
      <aside class="settings-nav">
        <button
          v-for="item in navigationSections"
          :key="item.key"
          type="button"
          class="nav-item"
          :class="{ active: activeSection === item.key }"
          @click="activeSection = item.key"
        >
          <el-icon><component :is="item.icon" /></el-icon>
          <span>{{ item.label }}</span>
        </button>
      </aside>

      <div class="settings-content">
        <section v-if="activeSection === 'library'" class="section-card">
          <div class="section-header">
            <div>
              <div class="eyebrow">Character Library</div>
              <h2>角色库管理</h2>
              <p>统一管理目标角色样本、封面与特征数据，为识别与跟踪任务提供稳定的身份库。</p>
            </div>
            <el-button type="primary" @click="openImportDrawer()">新增 / 扩展角色</el-button>
          </div>

          <div class="library-profile-bar">
            <div>
              <div class="profile-eyebrow">Library Profile</div>
              <div class="profile-title">当前角色库视图：{{ activeLibraryRecognitionLabel }}</div>
              <div class="profile-desc">不同特征模型对应独立的角色库视图，便于对照识别效果与样本配置。</div>
            </div>
            <div class="profile-actions">
              <el-button
                size="small"
                :type="isFollowingAnalysisRecognitionModel ? 'primary' : 'default'"
                @click="followCurrentAnalysisRecognitionModel"
              >
                跟随当前分析模型
              </el-button>
            </div>
          </div>

          <div class="profile-chip-group">
            <button
              v-for="item in options.recognition_models"
              :key="`library-profile-${item.key}`"
              type="button"
              class="profile-chip"
              :class="{ active: libraryPreferences.recognition_model_key === item.key }"
              @click="selectLibraryRecognitionModel(item.key)"
            >
              {{ item.label }}
            </button>
          </div>

          <div class="library-toolbar">
            <el-input
              v-model="libraryKeyword"
              clearable
              placeholder="搜索角色名"
              class="toolbar-search"
            />
            <el-select v-model="librarySourceFilter" clearable placeholder="来源筛选" class="toolbar-filter">
              <el-option label="本地上传" value="local_upload" />
              <el-option label="批量导入" value="batch_import" />
              <el-option label="历史视频" value="history_capture" />
              <el-option label="实验素材" value="web_import" />
            </el-select>
            <div class="toolbar-summary">
              共 {{ filteredCharacters.length }} 个角色 / {{ totalSampleCount }} 张样本
            </div>
          </div>

          <div class="library-grid" v-loading="charactersLoading">
            <div
              v-for="character in filteredCharacters"
              :key="character.name"
              class="character-card"
            >
              <img
                v-if="character.portrait_url"
                :src="toApiUrl(character.portrait_url)"
                :alt="character.name"
                class="character-cover"
              />
              <div v-else class="character-cover placeholder">{{ character.name.slice(0, 1) }}</div>

              <div class="character-body">
                <div class="character-title-row">
                  <strong>{{ character.name }}</strong>
                  <span class="sample-badge">{{ character.sample_count }} 张</span>
                </div>
                <div class="character-subtext">最近更新：{{ formatDate(character.updated_at) }}</div>
                <div class="character-cover-hint">
                  {{ character.portrait_url ? '封面已生成' : '未生成封面' }}
                </div>
                <div class="source-tags">
                  <span v-for="source in character.source_types" :key="source" class="source-tag">
                    {{ sourceLabelMap[source] || source }}
                  </span>
                </div>
              </div>

              <div class="card-actions">
                <el-button text @click="viewCharacter(character.name)">查看详情</el-button>
                <el-button text type="primary" @click="openImportDrawer(character.name)">补充样本</el-button>
                <el-dropdown trigger="click">
                  <el-button text>更多</el-button>
                  <template #dropdown>
                    <el-dropdown-menu>
                      <el-dropdown-item @click="rebuildCharacter(character.name)">重建特征</el-dropdown-item>
                      <el-dropdown-item divided @click="removeCharacter(character.name)">删除角色</el-dropdown-item>
                    </el-dropdown-menu>
                  </template>
                </el-dropdown>
              </div>
            </div>
          </div>

          <div v-if="filteredCharacters.length === 0 && !charactersLoading" class="empty-state">
            角色库还没有角色。可以导入本地图片、批量选择文件夹，或从历史视频中抓帧生成样本。
          </div>
        </section>

        <section v-else-if="activeSection === 'models'" class="section-card">
          <div class="section-header">
            <div>
              <div class="eyebrow">Models & Crop</div>
              <h2>模型与裁剪</h2>
              <p>配置角色样本裁剪、特征提取与跟踪参数，保证入库样本质量与模型匹配一致性。</p>
            </div>
            <el-button type="primary" plain @click="saveLibraryPreferences">保存默认配置</el-button>
          </div>

          <div class="form-grid">
            <div class="form-field">
              <label>默认 YOLO 模型</label>
              <el-select v-model="libraryPreferences.yolo_model_key" placeholder="选择 YOLO 模型">
                <el-option
                  v-for="item in options.yolo_models"
                  :key="item.key"
                  :label="item.label"
                  :value="item.key"
                />
              </el-select>
            </div>
            <div class="form-field">
              <label>默认特征模型</label>
              <el-select v-model="libraryPreferences.recognition_model_key" placeholder="选择特征模型">
                <el-option
                  v-for="item in options.recognition_models"
                  :key="item.key"
                  :label="item.label"
                  :value="item.key"
                />
              </el-select>
            </div>
            <div class="form-field">
              <label>默认 ByteTrack 配置</label>
              <el-select v-model="libraryPreferences.tracker_config_key" placeholder="选择跟踪配置">
                <el-option
                  v-for="item in options.tracker_configs"
                  :key="item.key"
                  :label="item.label"
                  :value="item.key"
                />
              </el-select>
            </div>
            <div class="form-field">
              <label>裁剪置信度阈值</label>
              <el-input-number v-model="libraryPreferences.confidence" :min="0.1" :max="0.95" :step="0.05" />
            </div>
            <div class="form-field">
              <label>最小目标尺寸</label>
              <el-input-number v-model="libraryPreferences.min_face_size" :min="16" :max="256" :step="4" />
            </div>
            <div class="form-field">
              <label>清晰度阈值</label>
              <el-input-number v-model="libraryPreferences.min_sharpness" :min="10" :max="500" :step="10" />
            </div>
            <div class="form-field">
              <label>最小样本数</label>
              <el-input-number v-model="libraryPreferences.min_samples" :min="1" :max="20" :step="1" />
            </div>
          </div>
        </section>

        <TrainingSection
          v-else-if="activeSection === 'training'"
          @showcase-refresh="handleShowcaseRefresh"
        />

        <ModelShowcaseSection
          v-else-if="activeSection === 'showcase'"
          :showcase-data="options.model_showcase"
          :analysis-model-options="analysisModelOptions"
          :analysis-model-selection="props.analysisModelSelection"
          @refresh="handleShowcaseRefresh"
          @model-selection-change="handleShowcaseModelSelection"
        />

        <section v-else-if="activeSection === 'export'" class="section-card">
          <div class="section-header">
            <div>
              <div class="eyebrow">Data Export</div>
              <h2>数据导出</h2>
              <p>自动生成实验统计、开放集案例与完整分析报告，便于论文结果整理与复现实验。</p>
            </div>
          </div>

          <div class="export-panel">
            <div class="export-main">
              <div class="export-icon">
                <el-icon><DataBoard /></el-icon>
              </div>
              <div>
                <div class="export-title">实验数据自动导出</div>
                <div class="export-desc">
                  输出位置：后端 <span>data/outputs/experiments</span>，包含时序平滑消融 CSV、开放集案例 CSV 和完整 JSON 报告。
                </div>
              </div>
            </div>
            <el-switch
              :model-value="props.dataExportEnabled"
              size="large"
              active-text="开启"
              inactive-text="关闭"
              @change="emit('data-export-change', $event)"
            />
          </div>

          <div class="export-grid">
            <div class="export-metric">
              <strong>实验 1</strong>
              <span>平滑前片段数、平滑后片段数、短片段过滤数、碎片降低率</span>
            </div>
            <div class="export-metric">
              <strong>实验 2</strong>
              <span>已知角色数、陌生人簇数，以及待人工核验的正确簇、误合并、误分裂字段</span>
            </div>
          </div>
        </section>

        <section v-else-if="activeSection === 'theme'" class="section-card">
          <div class="section-header">
            <div>
              <div class="eyebrow">Theme</div>
              <h2>界面主题</h2>
              <p>提供适合演示与长时间查看的浅色视觉方案，保持界面清晰统一。</p>
            </div>
          </div>

          <div class="theme-grid">
            <button
              v-for="theme in themes"
              :key="theme.key"
              type="button"
              class="theme-card"
              :class="{ active: props.currentTheme === theme.key }"
              @click="emit('theme-change', theme.key)"
            >
              <div class="theme-preview" :style="{ background: theme.gradient }">
                <span></span>
                <span></span>
                <span></span>
              </div>
              <div class="theme-name">{{ theme.label }}</div>
              <div class="theme-desc">{{ theme.description }}</div>
            </button>
          </div>
        </section>

        <section v-else class="section-card">
          <div class="section-header">
            <div>
              <div class="eyebrow">About</div>
              <h2>关于系统</h2>
              <p>系统围绕角色检测、身份识别、时序跟踪和结果展示构建完整的动漫视频分析流程。</p>
            </div>
          </div>

          <div class="about-grid">
            <div class="about-item">
              <strong>角色库存储方式</strong>
              <p>按模型视图组织原图、裁剪图、封面图、特征库与清单文件，保证样本来源可追踪。</p>
            </div>
            <div class="about-item">
              <strong>算法复用策略</strong>
              <p>结合 YOLO 目标检测、ResNet50 + ArcFace 特征提取与 ByteTrack 跟踪，实现检测到分析的闭环。</p>
            </div>
            <div class="about-item">
              <strong>后续扩展方向</strong>
              <p>系统支持继续扩展模型版本、实验展示和数据导出模块，便于后续评估与迭代。</p>
            </div>
          </div>
        </section>
      </div>
    </div>

    <el-drawer
      v-model="importDrawerVisible"
      size="68%"
      :destroy-on-close="false"
      class="import-drawer"
    >
      <template #header>
        <div class="drawer-header">
          <div>
            <h3>{{ importForm.character_name ? `维护角色：${importForm.character_name}` : '新增角色' }}</h3>
            <p>先生成裁剪预览，再确认哪些头像真正进入角色库。</p>
          </div>
        </div>
      </template>

      <el-steps :active="previewSessionId ? 3 : 1" finish-status="success" simple>
        <el-step title="角色与来源" />
        <el-step title="裁剪预览" />
        <el-step title="确认入库" />
      </el-steps>

      <div class="drawer-grid">
        <div class="panel-card">
          <div class="panel-title">1. 角色与素材来源</div>

          <div class="form-grid compact">
            <div class="form-field">
              <label>角色名</label>
              <el-input v-model="importForm.character_name" placeholder="例如 Hoshino_Ai" />
            </div>
            <div class="form-field">
              <label>裁剪模型</label>
              <el-select v-model="importForm.yolo_model_key" placeholder="选择 YOLO 模型">
                <el-option
                  v-for="item in options.yolo_models"
                  :key="item.key"
                  :label="item.label"
                  :value="item.key"
                />
              </el-select>
            </div>
          </div>

          <div class="source-switcher">
            <button
              v-for="item in sourceOptions"
              :key="item.key"
              type="button"
              class="source-chip"
              :class="{ active: importForm.source_type === item.key }"
              @click="importForm.source_type = item.key"
            >
              {{ item.label }}
            </button>
          </div>

          <div v-if="importForm.source_type === 'local_upload'" class="upload-zone">
            <input ref="localInputRef" type="file" multiple accept="image/*" hidden @change="handleLocalFiles" />
            <el-button type="primary" plain @click="localInputRef?.click()">选择本地图片</el-button>
            <div class="upload-hint">{{ selectedFilesText }}</div>
          </div>

          <div v-else-if="importForm.source_type === 'batch_import'" class="upload-zone">
            <input
              ref="folderInputRef"
              type="file"
              hidden
              multiple
              webkitdirectory
              directory
              @change="handleFolderFiles"
            />
            <el-button type="primary" plain @click="folderInputRef?.click()">选择图片文件夹</el-button>
            <div class="upload-hint">{{ selectedFilesText }}</div>
          </div>

          <div v-else-if="importForm.source_type === 'history_capture'" class="form-grid compact">
            <div class="form-field">
              <label>历史视频</label>
              <el-select v-model="importForm.history_video_id" placeholder="选择历史视频">
                <el-option
                  v-for="item in historyList"
                  :key="item.id"
                  :label="item.filename"
                  :value="item.id"
                />
              </el-select>
            </div>
            <div class="form-field">
              <label>抓帧间隔（秒）</label>
              <el-input-number v-model="importForm.capture_interval" :min="0.5" :max="20" :step="0.5" />
            </div>
            <div class="form-field">
              <label>最大抓帧数</label>
              <el-input-number v-model="importForm.max_images" :min="1" :max="24" :step="1" />
            </div>
          </div>

          <el-collapse class="experimental-collapse">
            <el-collapse-item title="实验素材来源" name="web">
              <div class="collapse-tip emphasis">{{ options.web_import_hint }}</div>
              <div class="form-grid compact">
                <div class="form-field">
                  <label>查询标签</label>
                  <el-input v-model="importForm.web_tag_query" placeholder="例如 hoshino_ai solo" />
                </div>
                <div class="form-field">
                  <label>抓取张数</label>
                  <el-input-number v-model="importForm.web_max_images" :min="1" :max="12" :step="1" />
                </div>
              </div>
              <div class="collapse-tip">实验素材仅作为样本补充来源，入库前仍需经过裁剪预览和人工确认。</div>
            </el-collapse-item>
          </el-collapse>

          <div class="panel-title secondary">2. 当前裁剪参数</div>
          <div class="form-grid compact">
            <div class="form-field">
              <label>置信度阈值</label>
              <el-input-number v-model="importForm.confidence" :min="0.1" :max="0.95" :step="0.05" />
            </div>
            <div class="form-field">
              <label>最小目标尺寸</label>
              <el-input-number v-model="importForm.min_face_size" :min="16" :max="256" :step="4" />
            </div>
            <div class="form-field">
              <label>清晰度阈值</label>
              <el-input-number v-model="importForm.min_sharpness" :min="10" :max="500" :step="10" />
            </div>
          </div>

          <div class="drawer-actions">
            <el-button type="primary" :loading="previewLoading" @click="generatePreview">生成裁剪预览</el-button>
          </div>
        </div>

        <div class="panel-card">
          <div class="panel-title">3. 裁剪预览与确认</div>
          <div v-if="previewItems.length === 0" class="empty-preview">
            还没有预览结果。先选择素材来源并生成裁剪预览。
          </div>

          <template v-else>
            <div class="preview-actions">
              <el-button text @click="selectAllPreviews">全选</el-button>
              <el-button text @click="keepTopPreviews">仅保留高质量样本</el-button>
              <span class="preview-summary">
                当前保留 {{ selectedPreviewIds.length }} / {{ previewItems.length }} 张
              </span>
            </div>
            <div class="preview-toolbar">
              <el-checkbox-group v-model="selectedPreviewIds">
                <div class="preview-grid">
                  <div
                    v-for="item in previewItems"
                    :key="item.id"
                    class="preview-card"
                  >
                    <el-checkbox :label="item.id" class="preview-check" />
                    <div class="preview-images">
                      <img :src="toApiUrl(item.original_url)" alt="original" />
                      <img :src="toApiUrl(item.cropped_url)" alt="cropped" />
                    </div>
                    <div class="preview-meta">
                      <span>质量 {{ item.quality_score }}</span>
                      <span>清晰度 {{ item.sharpness }}</span>
                      <span>置信度 {{ item.confidence }}</span>
                    </div>
                  </div>
                </div>
              </el-checkbox-group>
            </div>

            <div class="drawer-actions align-end">
              <el-button @click="resetPreview">重新选择</el-button>
              <el-button type="primary" :loading="confirmLoading" @click="confirmImport">
                确认加入角色库
              </el-button>
            </div>
          </template>
        </div>
      </div>
    </el-drawer>

    <el-dialog v-model="detailDialogVisible" width="72%" destroy-on-close>
      <template #header>
        <div class="drawer-header">
          <div>
            <h3>{{ detailCharacter?.name }}</h3>
            <p>角色封面、样本数量与已入库的裁剪头像。</p>
          </div>
        </div>
      </template>

      <div v-if="detailCharacter" class="detail-layout">
        <div class="detail-cover-card">
          <img
            v-if="detailCharacter.portrait_url"
            :src="toApiUrl(detailCharacter.portrait_url)"
            :alt="detailCharacter.name"
            class="detail-cover"
          />
          <div v-else class="detail-cover placeholder">{{ detailCharacter.name.slice(0, 1) }}</div>
          <div class="detail-name">{{ detailCharacter.name }}</div>
          <div class="detail-sample-count">{{ detailCharacter.sample_count }} 张样本</div>
          <div class="detail-meta-tags">
            <span
              v-for="source in detailCharacter.source_types"
              :key="source"
              class="source-tag"
            >
              {{ sourceLabelMap[source] || source }}
            </span>
          </div>
        </div>
        <div class="detail-images" v-if="detailCharacter.images?.length > 0">
          <img
            v-for="item in detailCharacter.images"
            :key="item.id"
            :src="toApiUrl(item.url)"
            :alt="item.id"
          />
        </div>
        <div class="detail-empty" v-else>
          该角色目前没有任何可供预览的裁切采样素材。
        </div>
      </div>
    </el-dialog>
  </section>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import {
  Brush,
  Collection,
  DataBoard,
  InfoFilled,
  Operation,
} from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import ModelShowcaseSection from '../components/settings/ModelShowcaseSection.vue'
import TrainingSection from '../components/settings/TrainingSection.vue'
import {
  apiDeleteLibraryCharacter,
  apiGetAnalysisModelOptions,
  apiGetHistory,
  apiGetLibraryCharacterDetail,
  apiGetLibraryCharacters,
  apiGetLibraryOptions,
  apiGetModelShowcase,
  apiImportLibraryCharacter,
  apiImportLibraryCharacterFromWeb,
  apiRebuildLibraryCharacter,
  toApiUrl,
} from '../api/index'

const LIBRARY_PREFS_KEY = 'act-library-prefs'

const props = defineProps({
  currentTheme: {
    type: String,
    default: 'moonlightBlue',
  },
  targetSection: {
    type: String,
    default: 'library',
  },
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

const emit = defineEmits(['theme-change', 'analysis-model-selection-change', 'data-export-change'])

const sections = [
  { key: 'library', label: '角色库', icon: Collection },
  { key: 'models', label: '模型与裁剪', icon: Operation },
  { key: 'export', label: '数据导出', icon: DataBoard },
  { key: 'theme', label: '界面主题', icon: Brush },
  { key: 'about', label: '关于系统', icon: InfoFilled },
]

const sourceLabelMap = {
  local_upload: '本地上传',
  batch_import: '批量导入',
  history_capture: '历史视频',
  web_import: '实验素材',
}

const sourceOptions = [
  { key: 'local_upload', label: '本地上传' },
  { key: 'batch_import', label: '批量导入' },
  { key: 'history_capture', label: '历史视频截图' },
]

const themes = [
  {
    key: 'moonlightBlue',
    label: '月白蓝',
    description: '冷静、清透，适合答辩投屏。',
    gradient: 'linear-gradient(135deg, #dfe9ff, #ebfffa)',
  },
  {
    key: 'mistGreen',
    label: '雾青绿',
    description: '更柔和，适合研究与整理感。',
    gradient: 'linear-gradient(135deg, #ddf5ef, #f5fbe3)',
  },
  {
    key: 'warmGold',
    label: '暖灰金',
    description: '温润但仍保持正式感。',
    gradient: 'linear-gradient(135deg, #f5ead9, #fff7e9)',
  },
]

const navigationSections = [
  { key: 'library', label: '角色库', icon: Collection },
  { key: 'models', label: '模型与裁剪', icon: Operation },
  { key: 'training', label: '模型训练', icon: DataBoard },
  { key: 'showcase', label: '模型展示', icon: DataBoard },
  { key: 'export', label: '数据导出', icon: DataBoard },
  { key: 'theme', label: '界面主题', icon: Brush },
  { key: 'about', label: '关于系统', icon: InfoFilled },
]

const activeSection = ref(props.targetSection || 'library')
const charactersLoading = ref(false)
const characters = ref([])
const historyList = ref([])
const libraryKeyword = ref('')
const librarySourceFilter = ref('')
const options = ref({
  yolo_models: [],
  recognition_models: [],
  tracker_configs: [],
  web_import_available: false,
  web_import_hint: '',
  model_showcase: {
    cards: [],
    experiment_groups: [],
  },
  crop_defaults: {
    confidence: 0.5,
    min_face_size: 30,
    min_sharpness: 100,
    min_samples: 3,
  },
})
const analysisModelOptions = ref({
  detection: { options: [], current_value: '' },
  recognition: { options: [], current_value: '' },
  tracker: { options: [], current_value: '' },
})

const importDrawerVisible = ref(false)
const detailDialogVisible = ref(false)
const previewLoading = ref(false)
const confirmLoading = ref(false)
const previewSessionId = ref('')
const previewContext = ref(null)
const previewItems = ref([])
const selectedPreviewIds = ref([])
const selectedFiles = ref([])
const detailCharacter = ref(null)

const localInputRef = ref(null)
const folderInputRef = ref(null)

const libraryPreferences = reactive({
  yolo_model_key: '',
  recognition_model_key: '',
  tracker_config_key: '',
  confidence: 0.5,
  min_face_size: 30,
  min_sharpness: 100,
  min_samples: 3,
})

const importForm = reactive({
  character_name: '',
  source_type: 'local_upload',
  yolo_model_key: '',
  history_video_id: null,
  capture_interval: 2,
  max_images: 10,
  web_tag_query: '',
  web_max_images: 8,
  confidence: 0.5,
  min_face_size: 30,
  min_sharpness: 100,
  min_samples: 3,
})

const selectedFilesText = computed(() => {
  if (selectedFiles.value.length === 0) {
    return '还没有选中图片素材。'
  }
  if (selectedFiles.value.length === 1) {
    return selectedFiles.value[0].name
  }
  return `已选择 ${selectedFiles.value.length} 个文件`
})

const filteredCharacters = computed(() => {
  return characters.value.filter((item) => {
    const matchesKeyword =
      !libraryKeyword.value ||
      item.name.toLowerCase().includes(libraryKeyword.value.trim().toLowerCase())
    const matchesSource =
      !librarySourceFilter.value ||
      (item.source_types || []).includes(librarySourceFilter.value)
    return matchesKeyword && matchesSource
  })
})

const totalSampleCount = computed(() => {
  return filteredCharacters.value.reduce((sum, item) => sum + (item.sample_count || 0), 0)
})

const activeLibraryRecognitionKey = computed(() => libraryPreferences.recognition_model_key || '')
const activeLibraryRecognitionLabel = computed(() => {
  return (
    options.value.recognition_models.find((item) => item.key === activeLibraryRecognitionKey.value)?.label ||
    '未选择特征模型'
  )
})
const analysisRecognitionModelKey = computed(() => {
  const targetPath = normalizeModelPath(props.analysisModelSelection?.recognition_model_path || '')
  if (!targetPath) {
    return ''
  }
  return (
    options.value.recognition_models.find((item) =>
      targetPath.endsWith(normalizeModelPath(item.key)),
    )?.key || ''
  )
})
const isFollowingAnalysisRecognitionModel = computed(() => {
  return !!analysisRecognitionModelKey.value && analysisRecognitionModelKey.value === activeLibraryRecognitionKey.value
})

const buildPreviewContext = () => ({
  characterName: importForm.character_name.trim(),
  sourceType: importForm.web_tag_query.trim() ? 'web_import' : importForm.source_type,
  recognitionModelKey: activeLibraryRecognitionKey.value || '',
  historyVideoId: importForm.history_video_id ?? null,
  webTagQuery: importForm.web_tag_query.trim(),
  yoloModelKey: importForm.yolo_model_key || '',
  confidence: importForm.confidence,
  minFaceSize: importForm.min_face_size,
  minSharpness: importForm.min_sharpness,
  minSamples: importForm.min_samples,
  selectedSignature: selectedFiles.value
    .map((file) => `${file.name}:${file.size}:${file.lastModified}`)
    .sort()
    .join('|'),
})

const isPreviewContextCurrent = () => {
  if (!previewContext.value) {
    return false
  }
  return JSON.stringify(previewContext.value) === JSON.stringify(buildPreviewContext())
}

const normalizeModelPath = (value) => String(value || '').replaceAll('/', '\\').toLowerCase()

const ensureValidPreferenceKey = (list, currentValue) => {
  if (list.some((item) => item.key === currentValue)) {
    return currentValue
  }
  return list[0]?.key || ''
}

const applyOptionsToPreferences = () => {
  const saved = localStorage.getItem(LIBRARY_PREFS_KEY)
  if (saved) {
    try {
      Object.assign(libraryPreferences, JSON.parse(saved))
    } catch (error) {
      console.warn('无法解析已保存的角色库偏好，已回退到默认值', error)
    }
  }
  const defaults = options.value.crop_defaults
  libraryPreferences.yolo_model_key = ensureValidPreferenceKey(
    options.value.yolo_models,
    libraryPreferences.yolo_model_key,
  )
  libraryPreferences.recognition_model_key = ensureValidPreferenceKey(
    options.value.recognition_models,
    libraryPreferences.recognition_model_key,
  )
  libraryPreferences.tracker_config_key = ensureValidPreferenceKey(
    options.value.tracker_configs,
    libraryPreferences.tracker_config_key,
  )
  libraryPreferences.confidence = libraryPreferences.confidence || defaults.confidence
  libraryPreferences.min_face_size = libraryPreferences.min_face_size || defaults.min_face_size
  libraryPreferences.min_sharpness = libraryPreferences.min_sharpness || defaults.min_sharpness
  libraryPreferences.min_samples = libraryPreferences.min_samples || defaults.min_samples
}

const syncImportDefaults = (characterName = '') => {
  importForm.character_name = characterName
  importForm.yolo_model_key = libraryPreferences.yolo_model_key
  importForm.confidence = libraryPreferences.confidence
  importForm.min_face_size = libraryPreferences.min_face_size
  importForm.min_sharpness = libraryPreferences.min_sharpness
  importForm.min_samples = libraryPreferences.min_samples
  importForm.source_type = 'local_upload'
  importForm.history_video_id = null
  importForm.capture_interval = 2
  importForm.max_images = 10
  importForm.web_tag_query = ''
  importForm.web_max_images = 8
}

const loadOptions = async () => {
  const response = await apiGetLibraryOptions(null)
  options.value = response.data
  applyOptionsToPreferences()
  syncImportDefaults(importForm.character_name)
}

const loadAnalysisModelOptions = async () => {
  const response = await apiGetAnalysisModelOptions()
  analysisModelOptions.value = response.data || {
    detection: { options: [], current_value: '' },
    recognition: { options: [], current_value: '' },
    tracker: { options: [], current_value: '' },
  }
}

const loadModelShowcase = async (modelSelection = props.analysisModelSelection) => {
  try {
    const response = await apiGetModelShowcase(modelSelection || null)
    options.value.model_showcase = response.data ?? { cards: [], experiment_groups: [] }
  } catch (error) {
    console.error('加载模型展示失败', error)
    options.value.model_showcase = { cards: [], experiment_groups: [] }
  }
}

const handleShowcaseRefresh = async () => {
  await Promise.all([loadOptions(), loadAnalysisModelOptions(), loadModelShowcase()])
}

const handleShowcaseModelSelection = (selection) => {
  emit('analysis-model-selection-change', selection)
}

const loadCharacters = async () => {
  charactersLoading.value = true
  try {
    const response = await apiGetLibraryCharacters(activeLibraryRecognitionKey.value || null)
    characters.value = response.data
  } finally {
    charactersLoading.value = false
  }
}

const loadHistory = async () => {
  const response = await apiGetHistory()
  historyList.value = response.data
}

const saveLibraryPreferences = () => {
  localStorage.setItem(LIBRARY_PREFS_KEY, JSON.stringify({ ...libraryPreferences }))
  syncImportDefaults(importForm.character_name)
  ElMessage.success('默认配置已保存')
}

const selectLibraryRecognitionModel = (modelKey) => {
  if (!modelKey || modelKey === libraryPreferences.recognition_model_key) {
    return
  }
  libraryPreferences.recognition_model_key = modelKey
}

const followCurrentAnalysisRecognitionModel = () => {
  if (!analysisRecognitionModelKey.value) {
    ElMessage.warning('当前分析链路还没有可映射的特征模型')
    return
  }
  if (analysisRecognitionModelKey.value === libraryPreferences.recognition_model_key) {
    ElMessage.success('角色库已经跟随当前分析模型')
    return
  }
  libraryPreferences.recognition_model_key = analysisRecognitionModelKey.value
  ElMessage.success('已切换到当前分析模型对应的角色库')
}

const openImportDrawer = (characterName = '') => {
  importDrawerVisible.value = true
  previewSessionId.value = ''
  previewContext.value = null
  previewItems.value = []
  selectedPreviewIds.value = []
  selectedFiles.value = []
  syncImportDefaults(characterName)
}

const resetPreview = () => {
  previewSessionId.value = ''
  previewContext.value = null
  previewItems.value = []
  selectedPreviewIds.value = []
}

const selectAllPreviews = () => {
  selectedPreviewIds.value = previewItems.value.map((item) => item.id)
}

const keepTopPreviews = () => {
  const keepCount = Math.max(importForm.min_samples, 3)
  selectedPreviewIds.value = previewItems.value.slice(0, keepCount).map((item) => item.id)
}

const handleLocalFiles = (event) => {
  selectedFiles.value = Array.from(event.target.files || [])
}

const handleFolderFiles = (event) => {
  selectedFiles.value = Array.from(event.target.files || [])
}

const generatePreview = async () => {
  if (!importForm.character_name.trim()) {
    ElMessage.warning('请先填写角色名')
    return
  }

  previewLoading.value = true
  try {
    let response
    if (importForm.web_tag_query.trim()) {
      const webForm = new FormData()
      webForm.append('character_name', importForm.character_name.trim())
      webForm.append('tag_query', importForm.web_tag_query.trim())
      webForm.append('max_images', String(importForm.web_max_images))
      webForm.append('yolo_model_key', importForm.yolo_model_key || '')
      webForm.append('confidence', String(importForm.confidence))
      webForm.append('min_face_size', String(importForm.min_face_size))
      webForm.append('min_sharpness', String(importForm.min_sharpness))
      webForm.append('min_samples', String(importForm.min_samples))
      webForm.append('recognition_model_key', activeLibraryRecognitionKey.value || '')
      response = await apiImportLibraryCharacterFromWeb(webForm)
    } else {
      const formData = new FormData()
      formData.append('character_name', importForm.character_name.trim())
      formData.append('source_type', importForm.source_type)
      formData.append('yolo_model_key', importForm.yolo_model_key || '')
      formData.append('finalize', 'false')
      formData.append('confidence', String(importForm.confidence))
      formData.append('min_face_size', String(importForm.min_face_size))
      formData.append('min_sharpness', String(importForm.min_sharpness))
      formData.append('min_samples', String(importForm.min_samples))
      formData.append('recognition_model_key', activeLibraryRecognitionKey.value || '')

      if (importForm.source_type === 'history_capture') {
        if (!importForm.history_video_id) {
          ElMessage.warning('请选择一个历史视频')
          previewLoading.value = false
          return
        }
        formData.append('history_video_id', String(importForm.history_video_id))
        formData.append('capture_interval', String(importForm.capture_interval))
        formData.append('max_images', String(importForm.max_images))
      } else {
        if (selectedFiles.value.length === 0) {
          ElMessage.warning('请先选择图片素材')
          previewLoading.value = false
          return
        }
        selectedFiles.value.forEach((file) => formData.append('files', file))
      }

      response = await apiImportLibraryCharacter(formData)
    }

    previewSessionId.value = response.data.session_id
    previewContext.value = buildPreviewContext()
    previewItems.value = response.data.previews
    selectedPreviewIds.value = response.data.previews.map((item) => item.id)
    ElMessage.success(response.data.message)
  } catch (error) {
    ElMessage.error(error?.response?.data?.detail || '生成预览失败')
  } finally {
    previewLoading.value = false
  }
}

const confirmImport = async () => {
  if (!previewSessionId.value || selectedPreviewIds.value.length === 0) {
    ElMessage.warning('请至少保留一个裁剪结果')
    return
  }

  if (!isPreviewContextCurrent()) {
    ElMessage.warning('当前角色信息或素材来源已变更，请重新生成预览')
    return
  }

  confirmLoading.value = true
  try {
    const formData = new FormData()
    formData.append('character_name', importForm.character_name.trim())
    formData.append('source_type', importForm.source_type)
    formData.append('finalize', 'true')
    formData.append('session_id', previewSessionId.value)
    formData.append('accepted_ids', JSON.stringify(selectedPreviewIds.value))
    formData.append('recognition_model_key', activeLibraryRecognitionKey.value || '')
    const response = await apiImportLibraryCharacter(formData)
    ElMessage.success(response.data.message)
    importDrawerVisible.value = false
    await loadCharacters()
    previewSessionId.value = ''
    previewContext.value = null
    previewItems.value = []
    selectedPreviewIds.value = []
  } catch (error) {
    ElMessage.error(error?.response?.data?.detail || '入库失败')
  } finally {
    confirmLoading.value = false
  }
}

const viewCharacter = async (name) => {
  const response = await apiGetLibraryCharacterDetail(name, activeLibraryRecognitionKey.value || null)
  detailCharacter.value = response.data
  detailDialogVisible.value = true
}

const rebuildCharacter = async (name) => {
  try {
    await apiRebuildLibraryCharacter(name, activeLibraryRecognitionKey.value || null)
    ElMessage.success('角色特征已重建')
    await loadCharacters()
  } catch (error) {
    ElMessage.error(error?.response?.data?.detail || '重建失败')
  }
}

const removeCharacter = async (name) => {
  try {
    await ElMessageBox.confirm(
      `这会删除角色 ${name} 的裁剪样本、封面和角色库记录，确定继续吗？`,
      '删除角色',
      {
        confirmButtonText: '确定删除',
        cancelButtonText: '取消',
        type: 'warning',
      },
    )
    await apiDeleteLibraryCharacter(name, activeLibraryRecognitionKey.value || null)
    ElMessage.success('角色已删除')
    await loadCharacters()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error(error?.response?.data?.detail || '删除失败')
    }
  }
}

const formatDate = (value) => {
  if (!value) {
    return '刚刚'
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`
}

watch(
  () => props.targetSection,
  (section) => {
    if (section) {
      activeSection.value = section
    }
  },
)

watch(
  () => props.analysisModelSelection,
  async (selection) => {
    await loadModelShowcase(selection)
  },
  { deep: true },
)

watch(
  () => libraryPreferences.recognition_model_key,
  async (nextValue, prevValue) => {
    if (!nextValue || nextValue === prevValue) {
      return
    }
    resetPreview()
    detailDialogVisible.value = false
    detailCharacter.value = null
    await loadCharacters()
  },
)

onMounted(async () => {
  await Promise.all([loadOptions(), loadAnalysisModelOptions()])
  await Promise.all([loadCharacters(), loadHistory(), loadModelShowcase()])
})
</script>

<style scoped>
.settings-page {
  height: 100%;
  min-height: 0;
  overflow: hidden;
  padding: 24px;
}

.settings-shell {
  height: 100%;
  min-height: 0;
  display: grid;
  grid-template-columns: 240px minmax(0, 1fr);
  gap: 18px;
}

.settings-nav,
.section-card,
.panel-card,
.detail-cover-card {
  border-radius: 12px;
  background: #ffffff;
  border: 1px solid var(--border-color);
  box-shadow: var(--shadow-light);
}

.settings-nav {
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  border-radius: 8px;
  border: none;
  background: transparent;
  color: var(--text-secondary);
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.nav-item.active,
.nav-item:hover {
  background: #f4f5f9;
  color: var(--text-primary);
}

.settings-content {
  min-width: 0;
  height: 100%;
  min-height: 0;
  display: flex;
  overflow: hidden;
}

.section-card {
  height: 100%;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  padding: 26px;
}

.section-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 22px;
}

.eyebrow {
  color: var(--accent-primary);
  font-size: 12px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  font-weight: 700;
}

.section-header h2 {
  margin: 8px 0 10px;
  font-size: 32px;
}

.section-header p {
  margin: 0;
  color: var(--text-secondary);
  max-width: 680px;
}

.library-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 16px;
}

.library-profile-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  padding: 14px 16px;
  margin-bottom: 14px;
  border: 1px solid var(--border-color);
  border-radius: 12px;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(244, 245, 249, 0.9));
}

.profile-eyebrow {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent-primary);
}

.profile-title {
  margin-top: 4px;
  font-size: 16px;
  font-weight: 700;
  color: var(--text-primary);
}

.profile-desc {
  margin-top: 6px;
  font-size: 13px;
  color: var(--text-secondary);
}

.profile-actions {
  flex-shrink: 0;
}

.profile-chip-group {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 18px;
}

.profile-chip {
  border: 1px solid var(--border-color);
  background: #fff;
  color: var(--text-secondary);
  border-radius: 999px;
  padding: 8px 14px;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.profile-chip:hover,
.profile-chip.active {
  border-color: var(--accent-primary);
  color: var(--accent-primary);
  background: rgba(16, 163, 127, 0.08);
}

.library-toolbar {
  display: grid;
  grid-template-columns: minmax(260px, 320px) 180px minmax(0, 1fr);
  gap: 14px;
  align-items: center;
  margin-bottom: 18px;
}

.toolbar-search,
.toolbar-filter {
  width: 100%;
}

.toolbar-summary {
  justify-self: end;
  color: var(--text-secondary);
  font-size: 13px;
}

.character-card {
  overflow: hidden;
  border-radius: 12px;
  background: #ffffff;
  border: 1px solid var(--border-color);
}

.character-cover {
  width: 100%;
  height: 160px;
  object-fit: cover;
  background: rgba(255, 255, 255, 0.74);
}

.character-cover.placeholder,
.detail-cover.placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--hero-gradient);
  color: var(--accent-primary);
  font-size: 48px;
  font-weight: 700;
}

.character-body {
  padding: 16px 16px 8px;
}

.character-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.sample-badge,
.source-tag {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  background: rgba(75, 120, 255, 0.10);
  color: var(--accent-primary);
}

.character-subtext {
  margin-top: 8px;
  color: var(--text-secondary);
  font-size: 12px;
}

.character-cover-hint {
  margin-top: 8px;
  color: var(--text-secondary);
  font-size: 12px;
}

.source-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

.card-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 12px 12px;
}

.empty-state,
.empty-preview {
  margin-top: 20px;
  padding: 24px;
  border-radius: 12px;
  background: #fcfcfc;
  border: 1px dashed var(--border-color);
  color: var(--text-secondary);
  line-height: 1.8;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}

.form-grid.compact {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 32px;
}

.form-field label {
  display: block;
  margin-bottom: 8px;
  font-size: 13px;
  color: var(--text-secondary);
}

.export-panel {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  padding: 22px;
  border-radius: 12px;
  background: #ffffff;
  border: 1px solid var(--border-color);
  box-shadow: var(--shadow-light);
}

.export-main {
  display: flex;
  align-items: center;
  gap: 16px;
  min-width: 0;
}

.export-icon {
  width: 44px;
  height: 44px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--accent-primary);
  background: rgba(16, 163, 127, 0.10);
  flex-shrink: 0;
}

.export-title {
  font-size: 18px;
  font-weight: 700;
  color: var(--text-primary);
}

.export-desc {
  margin-top: 6px;
  color: var(--text-secondary);
  font-size: 13px;
  line-height: 1.7;
}

.export-desc span {
  color: var(--accent-primary);
  font-weight: 700;
}

.export-grid {
  margin-top: 16px;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.export-metric {
  padding: 18px;
  border-radius: 12px;
  background: #f8fafc;
  border: 1px solid var(--border-color);
}

.export-metric strong,
.export-metric span {
  display: block;
}

.export-metric span {
  margin-top: 8px;
  color: var(--text-secondary);
  font-size: 13px;
  line-height: 1.7;
}

.theme-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
}

.showcase-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}

.showcase-card {
  padding: 22px;
  border-radius: 14px;
  background: #ffffff;
  border: 1px solid var(--border-color);
  box-shadow: var(--shadow-light);
}

.showcase-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 14px;
}

.showcase-label {
  font-size: 18px;
  font-weight: 700;
  color: var(--text-primary);
}

.showcase-artifact {
  display: inline-block;
  margin-top: 8px;
  color: var(--accent-primary);
  word-break: break-all;
}

.showcase-key {
  padding: 5px 10px;
  border-radius: 999px;
  background: rgba(75, 120, 255, 0.10);
  color: var(--accent-primary);
  font-size: 12px;
  font-weight: 700;
}

.showcase-summary {
  margin: 14px 0 0;
  color: var(--text-secondary);
  line-height: 1.8;
}

.showcase-block {
  margin-top: 18px;
}

.showcase-block-title {
  margin-bottom: 10px;
  font-size: 13px;
  font-weight: 700;
  color: var(--text-primary);
}

.showcase-list {
  margin: 0;
  padding-left: 18px;
  color: var(--text-secondary);
  line-height: 1.8;
}

.showcase-stat-grid {
  margin-top: 18px;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.showcase-stat-box {
  padding: 14px;
  border-radius: 12px;
  background: #f8fafc;
  border: 1px solid var(--border-color);
}

.showcase-stat-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.showcase-stat-item {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  color: var(--text-secondary);
  font-size: 13px;
}

.showcase-stat-item strong {
  color: var(--text-primary);
  text-align: right;
  word-break: break-all;
}

.theme-card {
  text-align: left;
  padding: 18px;
  border-radius: 12px;
  border: 1px solid var(--border-color);
  background: #ffffff;
  cursor: pointer;
  transition: transform 0.2s ease, border-color 0.2s ease;
}

.theme-card.active,
.theme-card:hover {
  transform: translateY(-2px);
  border-color: rgba(75, 120, 255, 0.35);
}

.theme-preview {
  height: 140px;
  border-radius: 18px;
  display: flex;
  align-items: flex-end;
  gap: 10px;
  padding: 14px;
}

.theme-preview span {
  flex: 1;
  height: 36px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.8);
}

.theme-name {
  margin-top: 14px;
  font-size: 16px;
  font-weight: 700;
}

.theme-desc {
  margin-top: 6px;
  color: var(--text-secondary);
  font-size: 13px;
}

.about-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
}

.about-item {
  padding: 20px;
  border-radius: 12px;
  background: #ffffff;
  border: 1px solid var(--border-color);
}

.about-item strong {
  display: block;
  margin-bottom: 10px;
}

.about-item p {
  margin: 0;
  color: var(--text-secondary);
  line-height: 1.8;
}

.drawer-header h3 {
  margin: 0;
  font-size: 24px;
}

.drawer-header p {
  margin: 8px 0 0;
  color: var(--text-secondary);
}

.drawer-grid {
  display: grid;
  grid-template-columns: minmax(320px, 1fr) minmax(0, 1fr);
  gap: 24px;
  margin-top: 20px;
}

.panel-card {
  padding: 20px;
  min-height: 0;
}

.panel-title {
  margin-bottom: 16px;
  font-size: 18px;
  font-weight: 700;
}

.panel-title.secondary {
  margin-top: 22px;
}

.source-switcher {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 18px;
}

.source-chip {
  padding: 10px 14px;
  border-radius: 8px;
  border: 1px solid var(--border-color);
  background: #ffffff;
  cursor: pointer;
}

.source-chip.active {
  background: rgba(75, 120, 255, 0.12);
  color: var(--accent-primary);
  border-color: rgba(75, 120, 255, 0.24);
}

.upload-zone {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 10px;
  margin-bottom: 18px;
}

.upload-hint,
.collapse-tip {
  color: var(--text-secondary);
  font-size: 12px;
  line-height: 1.7;
}

.collapse-tip.emphasis {
  margin-bottom: 12px;
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.72);
}

.experimental-collapse {
  margin: 8px 0 12px;
}

.drawer-actions {
  display: flex;
  justify-content: flex-start;
  gap: 12px;
  margin-top: 18px;
}

.drawer-actions.align-end {
  justify-content: flex-end;
}

.preview-actions {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.preview-toolbar {
  max-height: calc(100vh - 360px);
  overflow-y: auto;
  padding-right: 6px;
}

.preview-summary {
  margin-left: auto;
  color: var(--text-secondary);
  font-size: 12px;
}

.preview-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 14px;
}

.preview-card {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 14px;
  border-radius: 8px;
  background: #fdfdfd;
  border: 1px solid var(--border-color);
}

.preview-check {
  align-self: flex-start;
  margin-right: 0;
}

.preview-images {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin: 0;
}

.preview-images img {
  width: 100%;
  aspect-ratio: 1 / 1;
  object-fit: cover;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.82);
}

.preview-meta {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px 12px;
  color: var(--text-secondary);
  font-size: 12px;
  line-height: 1.5;
}

.preview-meta span {
  min-width: 0;
  word-break: break-word;
}

.detail-layout {
  display: grid;
  grid-template-columns: 260px minmax(0, 1fr);
  gap: 18px;
}

.detail-cover-card {
  padding: 18px;
  text-align: center;
}

.detail-meta-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
  margin-top: 12px;
}

.detail-cover {
  width: 100%;
  aspect-ratio: 1 / 1;
  object-fit: cover;
  border-radius: 22px;
  margin-bottom: 14px;
}

.detail-images {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 12px;
}

.detail-images img {
  width: 100%;
  aspect-ratio: 1 / 1;
  object-fit: cover;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.82);
}

.detail-name {
  margin-top: 14px;
  font-size: 18px;
  font-weight: 700;
  color: var(--text-primary);
}

.detail-sample-count {
  margin-top: 6px;
  font-size: 13px;
  color: var(--text-secondary);
}

.detail-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 12px;
  background: #fcfcfc;
  border: 1px dashed var(--border-color);
  color: var(--text-secondary);
  font-size: 14px;
  min-height: 240px;
}

@media (max-width: 1320px) {
  .settings-shell {
    grid-template-columns: 220px minmax(0, 1fr);
  }

  .theme-grid,
  .showcase-grid,
  .about-grid,
  .export-grid,
  .form-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .library-toolbar {
    grid-template-columns: 1fr;
  }

  .library-profile-bar {
    flex-direction: column;
    align-items: flex-start;
  }

  .export-panel {
    flex-direction: column;
    align-items: flex-start;
  }

  .drawer-grid,
  .detail-layout,
  .showcase-stat-grid {
    grid-template-columns: 1fr;
  }

  .preview-meta {
    grid-template-columns: 1fr;
  }
}
</style>
