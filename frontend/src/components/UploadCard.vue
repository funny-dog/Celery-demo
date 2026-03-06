<template>
  <div class="upload-card panel-card">
    <p class="panel-title">{{ panelTitle }}</p>
    
    <!-- Mode Switcher -->
    <div class="mode-switch">
      <button
        class="mode-btn"
        :class="{ active: mode === 'desensitize' }"
        :disabled="uploading"
        @click="$emit('update:mode', 'desensitize')"
      >
        CSV/XLSX 脱敏
      </button>
      <button
        class="mode-btn"
        :class="{ active: mode === 'split' }"
        :disabled="uploading"
        @click="$emit('update:mode', 'split')"
      >
        PDF/TXT 分片
      </button>
    </div>

    <!-- Backend Config -->
    <div class="api-config">
      <p class="api-title">后端地址（可选）</p>
      <div class="api-row">
        <input
          :value="apiBaseInput"
          @input="$emit('update:apiBaseInput', $event.target.value)"
          class="api-input"
          type="text"
          placeholder="留空表示同源"
          :disabled="uploading"
        />
        <button class="api-btn" :disabled="uploading" @click="$emit('saveApiBase')">保存</button>
        <button class="api-btn ghost" :disabled="uploading" @click="$emit('clearApiBase')">清空</button>
      </div>
      <p class="api-current">当前后端：{{ effectiveApiBaseLabel }}</p>
      <p v-if="apiBaseError" class="api-error">{{ apiBaseError }}</p>
    </div>

    <!-- Drag & Drop Zone -->
    <label 
      class="file-input"
      :class="{ 'drag-over': isDragOver }"
      @dragover.prevent="isDragOver = true"
      @dragleave.prevent="isDragOver = false"
      @drop.prevent="handleDrop"
    >
      <input
        type="file"
        :accept="fileAccept"
        @change="handleFileChange"
      />
      <div class="file-input-content">
        <svg class="upload-icon" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="17 8 12 3 7 8"></polyline>
          <line x1="12" y1="3" x2="12" y2="15"></line>
        </svg>
        <span class="file-hint-text">{{ fileName || fileHint }}</span>
      </div>
    </label>
    
    <!-- File Preview -->
    <div v-if="fileName" class="file-preview">
      <div class="file-info">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
        </svg>
        <span class="file-name">{{ fileName }}</span>
      </div>
      <button class="remove-btn" @click="$emit('removeFile')">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    </div>

    <!-- Submit Button -->
    <button 
      class="primary" 
      :disabled="!hasFile || uploading" 
      @click="$emit('submit')"
    >
      <svg v-if="uploading" class="spinner" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10" stroke-dasharray="60" stroke-dashoffset="20"></circle>
      </svg>
      {{ uploading ? '上传中...' : buttonLabel }}
    </button>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  mode: String,
  uploading: Boolean,
  apiBaseInput: String,
  effectiveApiBaseLabel: String,
  apiBaseError: String,
  fileName: String,
  hasFile: Boolean
})

const emit = defineEmits([
  'update:mode',
  'update:apiBaseInput',
  'saveApiBase',
  'clearApiBase',
  'submit',
  'removeFile',
  'fileChange'
])

const isDragOver = ref(false)

const fileAccept = computed(() => {
  return props.mode === 'split' 
    ? '.pdf,.txt,application/pdf,text/plain'
    : '.csv,.xlsx,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
})

const panelTitle = computed(() => {
  return props.mode === 'split' ? '上传 PDF/TXT 进行分片' : '上传 CSV/XLSX 进行脱敏'
})

const fileHint = computed(() => {
  return props.mode === 'split'
    ? '将 PDF/TXT 拖拽到这里，或点击选择文件'
    : '将 CSV/XLSX 拖拽到这里，或点击选择文件'
})

const buttonLabel = computed(() => {
  return props.mode === 'split' ? '上传并按 140MB 分片打包' : '上传并执行脱敏'
})

const handleDrop = (event) => {
  isDragOver.value = false
  const file = event.dataTransfer?.files?.[0]
  if (file) {
    emit('fileChange', file)
  }
}

const handleFileChange = (event) => {
  const file = event.target.files?.[0] || null
  emit('fileChange', file)
}
</script>

<style scoped>
.upload-card {
  display: flex;
  flex-direction: column;
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

.api-config {
  display: grid;
  gap: 8px;
  padding: 12px;
  border-radius: 12px;
  border: 1px solid var(--stroke);
  background: rgba(255, 255, 255, 0.7);
}

.api-title {
  margin: 0;
  font-size: 0.82rem;
  color: var(--ink-2);
}

.api-row {
  display: grid;
  grid-template-columns: 1fr auto auto;
  gap: 8px;
}

.api-input {
  border: 1px solid var(--stroke);
  border-radius: 8px;
  padding: 8px 10px;
  font-size: 0.88rem;
  min-width: 0;
}

.api-input:focus {
  outline: none;
  border-color: var(--accent-1);
  box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.1);
}

.api-btn {
  border: 1px solid var(--stroke);
  border-radius: 8px;
  background: white;
  padding: 8px 12px;
  font-size: 0.82rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.api-btn:hover:not(:disabled) {
  background: var(--accent-1);
  color: white;
  border-color: var(--accent-1);
}

.api-btn.ghost {
  color: var(--ink-2);
}

.api-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.api-current {
  margin: 0;
  color: var(--ink-2);
  font-size: 0.82rem;
}

.api-error {
  margin: 0;
  color: #b91c1c;
  font-size: 0.82rem;
}

.file-input {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  border-radius: 12px;
  border: 2px dashed rgba(31, 42, 48, 0.2);
  background: rgba(15, 118, 110, 0.04);
  cursor: pointer;
  transition: all 0.3s ease;
}

.file-input:hover {
  border-color: var(--accent-1);
  background: rgba(15, 118, 110, 0.08);
}

.file-input.drag-over {
  border-color: var(--accent-1);
  background: rgba(15, 118, 110, 0.12);
  transform: scale(1.01);
}

.file-input input {
  display: none;
}

.file-input-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.upload-icon {
  color: var(--accent-1);
  opacity: 0.7;
}

.file-hint-text {
  font-size: 0.9rem;
  color: var(--ink-2);
  text-align: center;
}

.file-preview {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px;
  background: rgba(15, 118, 110, 0.08);
  border-radius: 8px;
  animation: fadeIn 0.3s ease;
}

.file-info {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--ink-1);
}

.file-name {
  font-weight: 600;
  font-size: 0.9rem;
}

.remove-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: none;
  background: rgba(239, 68, 68, 0.1);
  color: #ef4444;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.remove-btn:hover {
  background: #ef4444;
  color: white;
}

.primary {
  border: none;
  border-radius: 12px;
  padding: 14px 20px;
  font-weight: 600;
  font-size: 1rem;
  background: linear-gradient(120deg, var(--accent-1), var(--accent-3));
  color: white;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
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

.primary:not(:disabled):active {
  transform: scale(0.98);
}

.spinner {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@media (max-width: 640px) {
  .api-row {
    grid-template-columns: 1fr;
  }
}
</style>
