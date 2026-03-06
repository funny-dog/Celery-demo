<template>
  <div class="rules-card panel-card">
    <p class="panel-title">{{ title }}</p>
    <p class="rules-desc">{{ description }}</p>
    
    <div class="rules-grid">
      <div v-for="rule in rules" :key="rule.label" class="rule-item">
        <span class="rule-label">{{ rule.label }}</span>
        <div class="rule-tags">
          <span v-for="tag in rule.tags" :key="tag" class="rule-tag">{{ tag }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  mode: {
    type: String,
    default: 'desensitize'
  }
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
      { 
        label: '支持格式', 
        icon: 'file',
        tags: ['.pdf', '.txt'] 
      },
      { 
        label: '分片大小', 
        icon: 'database',
        tags: ['每片最多 140MB'] 
      },
      { 
        label: '输出结果', 
        icon: 'archive',
        tags: ['ZIP 压缩包'] 
      }
    ]
  }
  
  return [
    { 
      label: '支持格式', 
      icon: 'file',
      tags: ['.csv', '.xlsx', '.json', '.jsonl'] 
    },
    { 
      label: '邮箱', 
      icon: 'mail',
      tags: ['email', 'e-mail', 'mail', '邮箱'] 
    },
    { 
      label: '手机号', 
      icon: 'phone',
      tags: ['phone', 'mobile', 'tel', 'telephone', '手机号', '电话'] 
    },
    { 
      label: '证件号', 
      icon: 'id-card',
      tags: ['id_card', 'idcard', 'ssn', 'passport', 'identity', '身份证', '证件'] 
    },
    { 
      label: '姓名', 
      icon: 'user',
      tags: ['name', 'full_name', 'first_name', 'last_name', '姓名'] 
    },
    { 
      label: '地址', 
      icon: 'map-pin',
      tags: ['address', 'addr', '地址'] 
    }
  ]
})
</script>

<style scoped>
.rules-card {
  grid-column: 1 / -1;
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

.rules-desc {
  margin: -4px 0 0;
  color: var(--ink-2);
  font-size: 0.95rem;
  line-height: 1.5;
}

.rules-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px 32px;
  padding-top: 12px;
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
  gap: 6px;
}

.rule-tag {
  font-size: 0.8rem;
  padding: 4px 10px;
  background: var(--bg-2);
  border: 1px solid rgba(31, 42, 48, 0.06);
  border-radius: 6px;
  color: var(--ink-1);
  font-family: 'Space Grotesk', monospace;
  transition: all 0.2s ease;
}

.rule-tag:hover {
  background: var(--accent-1);
  color: white;
  border-color: var(--accent-1);
}

@media (max-width: 640px) {
  .rules-grid {
    grid-template-columns: 1fr;
  }
}
</style>
