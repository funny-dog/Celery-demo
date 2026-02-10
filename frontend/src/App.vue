<template>
  <div class="page">
    <header class="hero">
      <div class="hero-text">
        <p class="eyebrow">脱敏与分片工具</p>
        <h1>上传文件，异步处理后下载结果。</h1>
        <p class="lead">
          同时支持 CSV/XLSX 脱敏与 PDF/TXT 分片。
          大文件会按 140MB 切分，并打包为 ZIP 下载。
        </p>
      </div>
      <div class="hero-card">
        <div>
          <p class="hero-label">技术栈</p>
          <p class="hero-value">FastAPI · Celery · Redis · PostgreSQL</p>
        </div>
        <div>
          <p class="hero-label">处理模式</p>
          <p class="hero-value">异步脱敏 / 140MB 分片 + ZIP</p>
        </div>
        <div>
          <p class="hero-label">刷新频率</p>
          <p class="hero-value">每 2 秒轮询一次状态</p>
        </div>
      </div>
    </header>

    <section class="panel">
      <div class="panel-card">
        <p class="panel-title">{{ panelTitle }}</p>
        <div class="mode-switch">
          <button
            class="mode-btn"
            :class="{ active: mode === 'desensitize' }"
            :disabled="uploading"
            @click="switchMode('desensitize')"
          >
            CSV/XLSX 脱敏
          </button>
          <button
            class="mode-btn"
            :class="{ active: mode === 'split' }"
            :disabled="uploading"
            @click="switchMode('split')"
          >
            PDF/TXT 分片
          </button>
        </div>
        <label class="file-input">
          <input
            type="file"
            :accept="fileAccept"
            @change="handleFile"
          />
          <span>选择文件</span>
        </label>
        <div class="file-meta">
          <p v-if="fileName" class="file-name">{{ fileName }}</p>
          <p v-else class="file-hint">{{ fileHint }}</p>
        </div>
        <button class="primary" :disabled="!selectedFile || uploading" @click="uploadFile">
          {{ uploading ? '上传中...' : uploadButtonLabel }}
        </button>
      </div>

      <div class="panel-card status">
        <div class="status-header">
          <div>
            <p class="panel-title">任务状态</p>
            <p class="status-state">{{ statusLabel }}</p>
          </div>
          <div class="status-actions">
            <p v-if="taskId" class="task-id">{{ taskId }}</p>
            <div class="button-group">
              <button
                v-if="['PENDING', 'PROGRESS'].includes(status.state)"
                @click="cancelTask"
                class="cancel-btn"
              >
                取消任务
              </button>
              <button v-if="status.state === 'SUCCESS'" @click="downloadResults" class="download-btn">
                下载结果
              </button>
            </div>
          </div>
        </div>

        <div class="progress-track">
          <div class="progress-bar" :style="{ width: progress + '%' }"></div>
        </div>

        <div class="status-meta">
          <span>{{ progress }}%</span>
          <span v-if="status.total">{{ status.current }}/{{ status.total }}</span>
          <span v-if="localizedStatusMessage">{{ localizedStatusMessage }}</span>
        </div>

        <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
      </div>

      <div v-if="mode === 'desensitize'" class="panel-card rules-card">
        <p class="panel-title">关键词识别规则</p>
        <p class="rules-desc">
          列名匹配以下关键词（不区分大小写）时会自动脱敏。
        </p>
        <div class="rules-grid">
          <div class="rule-item">
            <span class="rule-label">邮箱</span>
            <div class="rule-tags">
              <span>email</span><span>e-mail</span><span>mail</span><span>邮箱</span>
            </div>
          </div>
          <div class="rule-item">
            <span class="rule-label">手机号</span>
            <div class="rule-tags">
              <span>phone</span><span>mobile</span><span>tel</span><span>telephone</span><span>手机号</span><span>电话</span>
            </div>
          </div>
          <div class="rule-item">
            <span class="rule-label">证件号</span>
            <div class="rule-tags">
              <span>id_card</span><span>idcard</span><span>ssn</span><span>passport</span><span>identity</span><span>身份证</span><span>证件</span>
            </div>
          </div>
          <div class="rule-item">
            <span class="rule-label">姓名</span>
            <div class="rule-tags">
              <span>name</span><span>full_name</span><span>first_name</span><span>last_name</span><span>姓名</span>
            </div>
          </div>
          <div class="rule-item">
            <span class="rule-label">地址</span>
            <div class="rule-tags">
              <span>address</span><span>addr</span><span>地址</span>
            </div>
          </div>
        </div>
      </div>

      <div v-else class="panel-card rules-card">
        <p class="panel-title">PDF/TXT 分片规则</p>
        <p class="rules-desc">
          上传的 PDF/TXT 会被切分为多个分片（每片不超过 140MB），
          最终统一打包为一个 ZIP 供下载。
        </p>
        <div class="rules-grid">
          <div class="rule-item">
            <span class="rule-label">支持格式</span>
            <div class="rule-tags">
              <span>.pdf</span><span>.txt</span>
            </div>
          </div>
          <div class="rule-item">
            <span class="rule-label">分片大小</span>
            <div class="rule-tags">
              <span>每片最多 140MB</span>
            </div>
          </div>
          <div class="rule-item">
            <span class="rule-label">输出结果</span>
            <div class="rule-tags">
              <span>ZIP 压缩包</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, ref } from 'vue'

const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const selectedFile = ref(null)
const fileName = ref('')
const uploading = ref(false)
const taskId = ref('')
const errorMessage = ref('')
const mode = ref('desensitize')
const status = ref({
  state: 'IDLE',
  current: 0,
  total: 0,
  message: ''
})

const socket = ref(null)

const fileAccept = computed(() => {
  if (mode.value === 'split') {
    return '.pdf,.txt,application/pdf,text/plain'
  }
  return '.csv,.xlsx,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
})

const panelTitle = computed(() => {
  return mode.value === 'split' ? '上传 PDF/TXT 进行分片' : '上传 CSV/XLSX 进行脱敏'
})

const fileHint = computed(() => {
  return mode.value === 'split'
    ? '将 PDF/TXT 拖拽到这里，或点击选择文件。'
    : '将 CSV/XLSX 拖拽到这里，或点击选择文件。'
})

const uploadButtonLabel = computed(() => {
  return mode.value === 'split' ? '上传并按 140MB 分片打包' : '上传并执行脱敏'
})

const progress = computed(() => {
  if (!status.value.total) {
    return 0
  }
  const ratio = status.value.current / status.value.total
  return Math.min(100, Math.max(0, Math.round(ratio * 100)))
})

const statusLabel = computed(() => {
  if (status.value.state === 'IDLE') return '等待上传'
  if (status.value.state === 'PENDING') return '排队中'
  if (status.value.state === 'PROGRESS') return '处理中'
  if (status.value.state === 'SUCCESS') return '已完成'
  if (status.value.state === 'FAILURE') return '处理失败'
  if (status.value.state === 'REVOKED') return '已取消'
  return status.value.state
})

const localizeStatusMessage = (message) => {
  if (!message) return ''

  if (message === 'queued') return '排队中'
  if (message === 'splitting file') return '正在切分文件'
  if (message === 'completed') return '已完成'
  if (message.startsWith('completed (') && message.endsWith(')')) {
    const detail = message.slice('completed ('.length, -1)
    return `已完成（${detail}）`
  }

  const processRowMatch = message.match(/^Processing row (\d+)\/(\d+)$/)
  if (processRowMatch) {
    return `正在处理第 ${processRowMatch[1]} / ${processRowMatch[2]} 行`
  }

  const desensitizeMatch = message.match(/^Desensitizing row (\d+)\/(\d+)$/)
  if (desensitizeMatch) {
    return `正在脱敏第 ${desensitizeMatch[1]} / ${desensitizeMatch[2]} 行`
  }

  return message
}

const localizedStatusMessage = computed(() => localizeStatusMessage(status.value.message))

const switchMode = (nextMode) => {
  if (mode.value === nextMode) return
  mode.value = nextMode
  selectedFile.value = null
  fileName.value = ''
  errorMessage.value = ''
}

const handleFile = (event) => {
  const file = event.target.files?.[0] || null
  selectedFile.value = file
  fileName.value = file ? file.name : ''
}

const uploadFile = async () => {
  if (!selectedFile.value) {
    errorMessage.value =
      mode.value === 'split'
        ? '请先选择 PDF 或 TXT 文件。'
        : '请先选择 CSV 或 XLSX 文件。'
    return
  }

  errorMessage.value = ''
  uploading.value = true

  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)
    const endpoint = mode.value === 'split' ? '/upload/split' : '/upload/desensitize'

    const response = await fetch(`${apiBase}${endpoint}`, {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      const message = await response.text()
      throw new Error(message || '上传失败')
    }

    const data = await response.json()
    taskId.value = data.task_id
    status.value = {
      state: 'PENDING',
      current: 0,
      total: 0,
      message: 'queued'
    }

    connectWebSocket()
  } catch (error) {
    errorMessage.value = error?.message || '发生未知错误'
  } finally {
    uploading.value = false
  }
}

const connectWebSocket = () => {
  if (socket.value) {
    socket.value.close()
  }

  const wsBase = apiBase.replace(/^http/, 'ws')
  socket.value = new WebSocket(`${wsBase}/ws/status/${taskId.value}`)

  socket.value.onopen = () => {
    console.log('WS Connected')
  }

  socket.value.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      status.value.current = data.current
      status.value.total = data.total
      status.value.message = data.message

      if (typeof data.message === 'string' && data.message.startsWith('completed')) {
        status.value.state = 'SUCCESS'
        socket.value.close()
      } else {
        status.value.state = 'PROGRESS'
      }
    } catch (e) {
      console.error('WS parse error', e)
    }
  }
}

const cancelTask = async () => {
  if (!taskId.value) return
  try {
    await fetch(`${apiBase}/tasks/${taskId.value}/cancel`, { method: 'POST' })
    status.value.state = 'REVOKED'
    status.value.message = '已取消'
    if (socket.value) socket.value.close()
  } catch (e) {
    errorMessage.value = '取消任务失败'
  }
}

const downloadResults = () => {
  window.location.href = `${apiBase}/download/${taskId.value}`
}

onBeforeUnmount(() => {
  if (socket.value) socket.value.close()
})
</script>

<style scoped>
.download-btn {
  background: white;
  border: 1px solid var(--stroke);
  padding: 6px 12px;
  border-radius: 8px;
  font-size: 0.85rem;
  cursor: pointer;
  font-weight: 600;
  color: var(--accent-1);
  transition: all 0.2s;
  margin-left: auto;
}
.download-btn:hover {
  background: var(--accent-1);
  color: white;
  border-color: var(--accent-1);
}

.cancel-btn {
  background: white;
  border: 1px solid #ef4444;
  padding: 6px 12px;
  border-radius: 8px;
  font-size: 0.85rem;
  cursor: pointer;
  font-weight: 600;
  color: #ef4444;
  transition: all 0.2s;
}

.cancel-btn:hover {
  background: #ef4444;
  color: white;
}

.button-group {
  display: flex;
  gap: 8px;
}

.status-actions {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 8px;
}

.page {
  max-width: 1100px;
  margin: 0 auto;
  padding: 48px 24px 64px;
  display: flex;
  flex-direction: column;
  gap: 32px;
  animation: fadeIn 0.6s ease-out;
}

.hero {
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr);
  gap: 28px;
  align-items: center;
}

.hero-text h1 {
  font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
  font-size: clamp(2rem, 4vw, 3.25rem);
  margin: 0 0 12px;
}

.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.18em;
  font-weight: 600;
  font-size: 0.8rem;
  color: var(--accent-1);
  margin: 0 0 12px;
}

.lead {
  margin: 0;
  color: var(--ink-2);
  font-size: 1.05rem;
  line-height: 1.7;
}

.hero-card {
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid var(--stroke);
  border-radius: 20px;
  padding: 22px;
  display: grid;
  gap: 16px;
  box-shadow: var(--shadow);
}

.hero-label {
  margin: 0 0 6px;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--ink-2);
}

.hero-value {
  margin: 0;
  font-weight: 600;
  font-size: 1rem;
}

.panel {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 24px;
}

.panel-card {
  background: var(--card);
  border-radius: 20px;
  padding: 24px;
  border: 1px solid var(--stroke);
  box-shadow: 0 20px 40px rgba(31, 42, 48, 0.08);
  display: grid;
  gap: 16px;
}

.panel-title {
  margin: 0;
  font-weight: 600;
  font-size: 1rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--ink-2);
}

.mode-switch {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.mode-btn {
  border: 1px solid var(--stroke);
  background: white;
  color: var(--ink-1);
  border-radius: 999px;
  padding: 8px 14px;
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.mode-btn.active {
  background: var(--accent-1);
  color: white;
  border-color: var(--accent-1);
}

.file-input {
  display: inline-flex;
  align-items: center;
  gap: 12px;
  padding: 12px 18px;
  border-radius: 999px;
  border: 1px dashed rgba(31, 42, 48, 0.3);
  background: rgba(15, 118, 110, 0.08);
  cursor: pointer;
  width: fit-content;
}

.file-input input {
  display: none;
}

.file-input span {
  font-weight: 600;
  color: var(--accent-1);
}

.file-meta {
  color: var(--ink-2);
}

.file-name {
  margin: 0;
  font-weight: 600;
}

.file-hint {
  margin: 0;
}

.primary {
  border: none;
  border-radius: 12px;
  padding: 12px 18px;
  font-weight: 600;
  background: linear-gradient(120deg, var(--accent-1), var(--accent-3));
  color: white;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.primary:not(:disabled):hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(15, 118, 110, 0.25);
}

.status {
  display: grid;
  gap: 18px;
}

.status-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 16px;
}

.status-state {
  margin: 6px 0 0;
  font-size: 1.1rem;
  font-weight: 600;
}

.task-id {
  margin: 0;
  font-size: 0.85rem;
  color: var(--ink-2);
}

.progress-track {
  height: 14px;
  background: rgba(31, 42, 48, 0.08);
  border-radius: 999px;
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  background: linear-gradient(90deg, var(--accent-2), var(--accent-1));
  transition: width 0.4s ease;
}

.status-meta {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  color: var(--ink-2);
  font-size: 0.95rem;
}

.error {
  margin: 0;
  color: #b91c1c;
  font-weight: 600;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(12px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (max-width: 900px) {
  .hero {
    grid-template-columns: 1fr;
  }

  .panel {
    grid-template-columns: 1fr;
  }
}

.rules-card {
  grid-column: 1 / -1;
  gap: 20px;
}

.rules-desc {
  margin: -8px 0 0;
  color: var(--ink-2);
  font-size: 0.95rem;
  line-height: 1.5;
}

.rules-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 24px 32px;
  padding-top: 8px;
  border-top: 1px solid var(--stroke);
}

.rule-item {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.rule-label {
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-weight: 600;
  color: var(--accent-1);
}

.rule-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.rule-tags span {
  font-size: 0.85rem;
  padding: 4px 10px;
  background: var(--bg-2);
  border: 1px solid rgba(31, 42, 48, 0.06);
  border-radius: 6px;
  color: var(--ink-1);
  font-family: 'Space Grotesk', monospace;
}
</style>
