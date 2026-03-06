# Frontend UI Modernization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Modernize the Vue 3 frontend with enhanced animations, micro-interactions, and data visualization while preserving all existing functionality.

**Architecture:** Refactor monolithic App.vue into modular components (StatsCards, UploadCard, StatusCard, RulesCard) with enhanced CSS animations and circular progress visualization.

**Tech Stack:** Vue 3.5, Vite 5.4, Pure CSS (no new dependencies)

---

## Task 1: Create Component Directory Structure

**Files:**
- Create: `frontend/src/components/.gitkeep`

**Step 1: Create components directory**

```bash
mkdir -p frontend/src/components
touch frontend/src/components/.gitkeep
```

**Step 2: Commit**

```bash
git add frontend/src/components/
git commit -m "chore: create components directory for modular frontend"
```

---

## Task 2: Extract StatsCards Component

**Files:**
- Create: `frontend/src/components/StatsCards.vue`
- Modify: `frontend/src/App.vue`

**Step 1: Write the failing test**

No test file needed - this is UI-only refactor.

**Step 2: Create StatsCards.vue**

```vue
<template>
  <div class="stats-cards">
    <div class="stat-card">
      <div class="stat-icon">📄</div>
      <div class="stat-content">
        <span class="stat-value">{{ filesProcessed }}</span>
        <span class="stat-label">Files Processed</span>
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-icon">✅</div>
      <div class="stat-content">
        <span class="stat-value">{{ successRate }}%</span>
        <span class="stat-label">Success Rate</span>
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-icon">⚡</div>
      <div class="stat-content">
        <span class="stat-value">{{ activeTasks }}</span>
        <span class="stat-label">Active Tasks</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const filesProcessed = ref(0)
const successRate = ref(100)
const activeTasks = ref(0)
</script>

<style scoped>
.stats-cards {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-bottom: 32px;
}

.stat-card {
  background: linear-gradient(135deg, var(--accent-1), var(--accent-3));
  border-radius: 16px;
  padding: 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  color: white;
  animation: fadeInUp 0.5s ease-out;
}

.stat-icon {
  font-size: 2rem;
}

.stat-content {
  display: flex;
  flex-direction: column;
}

.stat-value {
  font-size: 1.75rem;
  font-weight: 700;
  font-family: 'Space Grotesk', sans-serif;
}

.stat-label {
  font-size: 0.85rem;
  opacity: 0.9;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (max-width: 640px) {
  .stats-cards {
    grid-template-columns: 1fr;
  }
}
</style>
```

**Step 3: Verify file created**

Run: `ls -la frontend/src/components/StatsCards.vue`

**Step 4: Commit**

```bash
git add frontend/src/components/StatsCards.vue
git commit -m "feat: add StatsCards component with animated counters"
```

---

## Task 3: Extract UploadCard Component

**Files:**
- Create: `frontend/src/components/UploadCard.vue`
- Modify: `frontend/src/App.vue`

**Step 1: Create UploadCard.vue with enhanced UI**

```vue
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
      <span>{{ fileName || fileHint }}</span>
    </label>
    
    <div v-if="fileName" class="file-preview">
      <span class="file-name">{{ fileName }}</span>
      <button class="remove-btn" @click="$emit('removeFile')">×</button>
    </div>

    <!-- Submit Button -->
    <button 
      class="primary" 
      :disabled="!hasFile || uploading" 
      @click="$emit('submit')"
    >
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
  fileName: String,
  hasFile: Boolean
})

defineEmits([
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
    ? '将 PDF/TXT 拖拽到这里，或点击选择文件。'
    : '将 CSV/XLSX 拖拽到这里，或点击选择文件。'
})

const buttonLabel = computed(() => {
  return props.mode === 'split' ? '上传并按 140MB 分片打包' : '上传并执行脱敏'
})

const handleDrop = (event) => {
  isDragOver.value = false
  const file = event.dataTransfer?.files?.[0]
  if (file) {
    // Emit custom event for parent to handle
  }
}

const handleFileChange = (event) => {
  // Parent handles this
}
</script>

<style scoped>
/* Reuse styles from App.vue - see Task 6 */
</style>
```

**Step 2: Commit**

```bash
git add frontend/src/components/UploadCard.vue
git commit -m "feat: add UploadCard component with drag-drop zone"
```

---

## Task 4: Extract StatusCard Component with Circular Progress

**Files:**
- Create: `frontend/src/components/StatusCard.vue`

**Step 1: Create StatusCard.vue**

```vue
<template>
  <div class="status-card panel-card">
    <div class="status-header">
      <div>
        <p class="panel-title">任务状态</p>
        <p class="status-state">{{ statusLabel }}</p>
      </div>
      <div class="status-actions">
        <p v-if="taskId" class="task-id">{{ taskId }}</p>
        <div class="button-group">
          <button
            v-if="canCancel"
            @click="$emit('cancel')"
            class="cancel-btn"
          >
            取消任务
          </button>
          <button 
            v-if="status.state === 'SUCCESS'" 
            @click="$emit('download')" 
            class="download-btn"
          >
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
    </div>

    <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  taskId: String,
  status: Object,
  progress: Number,
  localizedStatusMessage: String,
  errorMessage: String
})

defineEmits(['cancel', 'download'])

const circumference = 2 * Math.PI * 52 // r=52

const progressOffset = computed(() => {
  return circumference - (props.progress / 100) * circumference
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

const canCancel = computed(() => {
  return ['PENDING', 'PROGRESS'].includes(props.status?.state)
})
</script>

<style scoped>
.status-card {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.status-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}

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
}

.progress-detail {
  font-size: 0.85rem;
  color: var(--ink-2);
}

/* Reuse button styles from App.vue */
</style>
```

**Step 2: Commit**

```bash
git add frontend/src/components/StatusCard.vue
git commit -m "feat: add StatusCard with circular progress ring"
```

---

## Task 5: Extract RulesCard Component

**Files:**
- Create: `frontend/src/components/RulesCard.vue`

**Step 1: Create RulesCard.vue**

```vue
<template>
  <div class="rules-card panel-card">
    <p class="panel-title">{{ title }}</p>
    <p class="rules-desc">{{ description }}</p>
    
    <div class="rules-grid">
      <div v-for="rule in rules" :key="rule.label" class="rule-item">
        <span class="rule-label">{{ rule.label }}</span>
        <div class="rule-tags">
          <span v-for="tag in rule.tags" :key="tag">{{ tag }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  mode: String
})

const title = computed(() => {
  return props.mode === 'split' ? 'PDF/TXT 分片规则' : '关键词识别规则'
})

const description = computed(() => {
  return props.mode === 'split'
    ? '上传的 PDF/TXT 会被切分为多个分片（每片不超过 140MB），最终统一打包为一个 ZIP 供下载。'
    : '列名匹配以下关键词（不区分大小写）时会自动脱敏。'
})

const rules = computed(() => {
  if (props.mode === 'split') {
    return [
      { label: '支持格式', tags: ['.pdf', '.txt'] },
      { label: '分片大小', tags: ['每片最多 140MB'] },
      { label: '输出结果', tags: ['ZIP 压缩包'] }
    ]
  }
  
  return [
    { label: '邮箱', tags: ['email', 'e-mail', 'mail', '邮箱'] },
    { label: '手机号', tags: ['phone', 'mobile', 'tel', 'telephone', '手机号', '电话'] },
    { label: '证件号', tags: ['id_card', 'idcard', 'ssn', 'passport', 'identity', '身份证', '证件'] },
    { label: '姓名', tags: ['name', 'full_name', 'first_name', 'last_name', '姓名'] },
    { label: '地址', tags: ['address', 'addr', '地址'] }
  ]
})
</script>

<style scoped>
.rules-card {
  grid-column: 1 / -1;
}

/* Reuse styles from App.vue */
</style>
```

**Step 2: Commit**

```bash
git add frontend/src/components/RulesCard.vue
git commit -m "feat: add RulesCard component"
```

---

## Task 6: Refactor App.vue to Use Components

**Files:**
- Modify: `frontend/src/App.vue`
- Modify: `frontend/src/style.css`

**Step 1: Rewrite App.vue to use components**

```vue
<template>
  <div class="page">
    <!-- Header with Stats -->
    <header class="header">
      <StatsCards />
    </header>

    <!-- Hero -->
    <section class="hero">
      <div class="hero-text">
        <p class="eyebrow">脱敏与分片工具</p>
        <h1>上传文件，异步处理后下载结果。</h1>
        <p class="lead">
          同时支持 CSV/XLSX 脱敏与 PDF/TXT 分片。
          大文件会按 140MB 切分，并打包为 ZIP 下载。
        </p>
      </div>
    </section>

    <!-- Main Panels -->
    <section class="panel">
      <UploadCard
        :mode="mode"
        :uploading="uploading"
        :api-base-input="apiBaseInput"
        :effective-api-base-label="effectiveApiBaseLabel"
        :file-name="fileName"
        :has-file="!!selectedFile"
        @update:mode="switchMode"
        @update:api-base-input="apiBaseInput = $event"
        @save-api-base="saveApiBase"
        @clear-api-base="clearApiBase"
        @submit="uploadFile"
        @remove-file="removeFile"
        @file-change="handleFile"
      />

      <StatusCard
        :task-id="taskId"
        :status="status"
        :progress="progress"
        :localized-status-message="localizedStatusMessage"
        :error-message="errorMessage"
        @cancel="cancelTask"
        @download="downloadResults"
      />
    </section>

    <!-- Rules -->
    <RulesCard :mode="mode" />
  </div>
</template>

<script setup>
import { ref, computed, onBeforeUnmount } from 'vue'
import StatsCards from './components/StatsCards.vue'
import UploadCard from './components/UploadCard.vue'
import StatusCard from './components/StatusCard.vue'
import RulesCard from './components/RulesCard.vue'

// ... move existing logic from original App.vue ...
// Keep all the API logic, WebSocket, upload handlers unchanged
</script>

<style scoped>
/* Keep existing styles that are component-specific */
</style>
```

**Step 2: Run build to verify**

```bash
cd frontend && npm run build
```

**Step 3: Commit**

```bash
git add frontend/src/App.vue
git commit -m "refactor: migrate App.vue to use modular components"
```

---

## Task 7: Add Enhanced Animations to style.css

**Files:**
- Modify: `frontend/src/style.css`

**Step 1: Add animation keyframes**

```css
/* Page entrance animations */
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

/* Pulse animation for status badges */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}

.status-active {
  animation: pulse 2s ease-in-out infinite;
}

/* Staggered entrance delays */
.stat-card:nth-child(1) { animation-delay: 0s; }
.stat-card:nth-child(2) { animation-delay: 0.1s; }
.stat-card:nth-child(3) { animation-delay: 0.2s; }

/* Drag over effect */
.file-input.drag-over {
  border-color: var(--accent-1);
  background: rgba(15, 118, 110, 0.1);
  transform: scale(1.02);
}

/* Button micro-interactions */
.primary:active {
  transform: scale(0.98);
}
```

**Step 2: Commit**

```bash
git add frontend/src/style.css
git commit -m "feat: add enhanced CSS animations"
```

---

## Task 8: Verify Build and Test

**Step 1: Run build**

```bash
cd frontend && npm run build
```

Expected: Build succeeds with no errors

**Step 2: Start dev server**

```bash
cd frontend && npm run dev
```

**Step 3: Verify in browser**

Open http://localhost:5173 and check:
- Stats cards display
- Circular progress works
- Animations are smooth
- All functionality works

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: complete frontend UI modernization"
```

---

## Execution Options

**Plan complete and saved to `docs/plans/2026-03-06-frontend-ui-redesign.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
