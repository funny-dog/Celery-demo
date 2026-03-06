<template>
  <div class="status-card panel-card">
    <div class="status-header">
      <div>
        <p class="panel-title">任务状态</p>
        <p class="status-state" :class="statusClass">{{ statusLabel }}</p>
      </div>
      <div class="status-actions">
        <p v-if="taskId" class="task-id" :title="taskId">{{ truncatedTaskId }}</p>
        <div class="button-group">
          <button
            v-if="canCancel"
            @click="$emit('cancel')"
            class="cancel-btn"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
            取消任务
          </button>
          <button 
            v-if="status.state === 'SUCCESS'" 
            @click="$emit('download')" 
            class="download-btn"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="7 10 12 15 17 10"></polyline>
              <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
            下载结果
          </button>
        </div>
      </div>
    </div>

    <!-- Circular Progress -->
    <div class="progress-ring-container">
      <svg class="progress-ring" viewBox="0 0 120 120">
        <circle
          class="progress-ring-bg"
          cx="60"
          cy="60"
          r="52"
          fill="none"
          stroke-width="8"
        />
        <circle
          class="progress-ring-fill"
          :class="{ 'is-success': status.state === 'SUCCESS' }"
          cx="60"
          cy="60"
          r="52"
          fill="none"
          stroke-width="8"
          :stroke-dasharray="circumference"
          :stroke-dashoffset="progressOffset"
        />
      </svg>
      <div class="progress-text">
        <span class="progress-percent">{{ progress }}%</span>
        <span v-if="status.total" class="progress-detail">
          {{ status.current }}/{{ status.total }}
        </span>
      </div>
    </div>

    <!-- Status Message -->
    <div class="status-meta">
      <span v-if="localizedStatusMessage">{{ localizedStatusMessage }}</span>
      <span v-else-if="!taskId" class="status-idle">等待上传文件</span>
    </div>

    <p v-if="errorMessage" class="error">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="12" y1="8" x2="12" y2="12"></line>
        <line x1="12" y1="16" x2="12.01" y2="16"></line>
      </svg>
      {{ errorMessage }}
    </p>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  taskId: String,
  status: {
    type: Object,
    default: () => ({ state: 'IDLE', current: 0, total: 0, message: '' })
  },
  progress: {
    type: Number,
    default: 0
  },
  localizedStatusMessage: String,
  errorMessage: String
})

defineEmits(['cancel', 'download'])

const circumference = 2 * Math.PI * 52 // r=52

const progressOffset = computed(() => {
  return circumference - (props.progress / 100) * circumference
})

const truncatedTaskId = computed(() => {
  if (!props.taskId) return ''
  return props.taskId.length > 12 
    ? props.taskId.slice(0, 8) + '...' + props.taskId.slice(-4)
    : props.taskId
})

const statusLabel = computed(() => {
  const state = props.status?.state
  if (state === 'IDLE') return '等待上传'
  if (state === 'PENDING') return '排队中'
  if (state === 'PROGRESS') return '处理中'
  if (state === 'SUCCESS') return '已完成'
  if (state === 'FAILURE') return '处理失败'
  if (state === 'REVOKED') return '已取消'
  return state
})

const statusClass = computed(() => {
  const state = props.status?.state
  if (state === 'PENDING' || state === 'PROGRESS') return 'status-active'
  if (state === 'SUCCESS') return 'status-success'
  if (state === 'FAILURE') return 'status-error'
  if (state === 'REVOKED') return 'status-cancelled'
  return ''
})

const canCancel = computed(() => {
  return ['PENDING', 'PROGRESS'].includes(props.status?.state)
})
</script>

<style scoped>
.status-card {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.status-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}

.panel-title {
  margin: 0;
  font-weight: 600;
  font-size: 1rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--ink-2);
}

.status-state {
  margin: 6px 0 0;
  font-size: 1.1rem;
  font-weight: 600;
  transition: color 0.3s ease;
}

.status-active {
  color: var(--accent-1);
}

.status-success {
  color: #22c55e;
}

.status-error {
  color: #ef4444;
}

.status-cancelled {
  color: #f59e0b;
}

.status-actions {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 8px;
}

.task-id {
  margin: 0;
  font-size: 0.8rem;
  font-family: 'Space Grotesk', monospace;
  color: var(--ink-2);
  background: rgba(31, 42, 48, 0.06);
  padding: 4px 8px;
  border-radius: 4px;
}

.button-group {
  display: flex;
  gap: 8px;
}

.download-btn, .cancel-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.download-btn {
  background: white;
  border: 1px solid var(--stroke);
  color: var(--accent-1);
}

.download-btn:hover {
  background: var(--accent-1);
  color: white;
  border-color: var(--accent-1);
}

.cancel-btn {
  background: white;
  border: 1px solid #ef4444;
  color: #ef4444;
}

.cancel-btn:hover {
  background: #ef4444;
  color: white;
}

/* Circular Progress */
.progress-ring-container {
  position: relative;
  width: 160px;
  height: 160px;
  margin: 0 auto;
}

.progress-ring {
  transform: rotate(-90deg);
  width: 100%;
  height: 100%;
}

.progress-ring-bg {
  stroke: rgba(31, 42, 48, 0.08);
}

.progress-ring-fill {
  stroke: var(--accent-1);
  stroke-linecap: round;
  transition: stroke-dashoffset 0.4s ease;
}

.progress-ring-fill.is-success {
  stroke: #22c55e;
}

.progress-text {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
}

.progress-percent {
  display: block;
  font-size: 2rem;
  font-weight: 700;
  font-family: 'Space Grotesk', sans-serif;
  color: var(--ink-1);
  line-height: 1;
}

.progress-detail {
  display: block;
  font-size: 0.85rem;
  color: var(--ink-2);
  margin-top: 4px;
}

.status-meta {
  display: flex;
  justify-content: center;
  color: var(--ink-2);
  font-size: 0.95rem;
  min-height: 24px;
}

.status-idle {
  opacity: 0.5;
}

.error {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0;
  color: #ef4444;
  font-weight: 600;
  font-size: 0.9rem;
  padding: 12px;
  background: rgba(239, 68, 68, 0.08);
  border-radius: 8px;
}

/* Pulse animation for active status */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}

.status-active {
  animation: pulse 2s ease-in-out infinite;
}
</style>
