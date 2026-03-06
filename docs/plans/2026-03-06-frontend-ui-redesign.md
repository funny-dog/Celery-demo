# Frontend UI Modernization Design

**Date:** 2026-03-06  
**Project:** Bulk Desensitizer Frontend Redesign  
**Status:** Approved

## 1. Overview

Modernize the existing Vue 3 frontend with enhanced animations, micro-interactions, and data visualization while preserving all existing functionality.

## 2. Functionality (Unchanged)

All current features remain:
- CSV/XLSX desensitization mode
- PDF/TXT splitting mode
- Task status polling with WebSocket
- Progress bar display
- Backend address configuration
- Download/Cancel task actions

## 3. UI/UX Design

### 3.1 Layout Structure

```
┌─────────────────────────────────────────────────────┐
│  Header: Logo + Stats Cards Row                     │
├─────────────────────────────────────────────────────┤
│  Hero: Title + Description                          │
├──────────────────────────┬──────────────────────────┤
│  Upload Card             │  Status Card              │
│  - Mode Switcher         │  - Task Info              │
│  - Drag & Drop Zone      │  - Circular Progress     │
│  - File Preview          │  - Action Buttons         │
│  - Backend Config        │  - Status Badge          │
├──────────────────────────┴──────────────────────────┤
│  Rules Card (Full Width)                            │
└─────────────────────────────────────────────────────┘
```

### 3.2 Visual Design

**Color Palette:**
- Primary: `#0f766e` (teal-700)
- Primary Light: `#14b8a6` (teal-500)
- Primary Dark: `#0d9488` (teal-600)
- Accent: `#f59e0b` (amber-500)
- Success: `#22c55e` (green-500)
- Error: `#ef4444` (red-500)
- Background: `#f8fafc` (slate-50)
- Card: `#ffffff`
- Text Primary: `#1e293b` (slate-800)
- Text Secondary: `#64748b` (slate-500)

**Typography:**
- Headings: Space Grotesk, system-ui
- Body: system-ui, sans-serif
- Monospace: JetBrains Mono (for badges/tags)

**Spacing:**
- Base unit: 4px
- Card padding: 24px
- Section gap: 32px
- Border radius: 16px (cards), 8px (buttons)

### 3.3 Component Designs

#### Stats Cards (Header)
- 3 cards: Files Processed, Success Rate, Active Tasks
- Icon + number + label
- Subtle gradient background
- Count-up animation on load

#### Upload Card
- Mode toggle pills at top
- Drag & drop zone with dashed border
- File icon + name + size on selection
- Remove button (X) on hover
- Backend URL input with save/clear buttons
- Submit button with gradient background

#### Status Card
- Circular progress ring (SVG-based)
- Animated fill on progress
- Status badge with pulse animation for PENDING/PROGRESS
- Task ID display (truncated)
- Download/Cancel buttons with icons

#### Rules Card
- Grid of rule categories
- Each with icon, label, and tag pills
- Hover effect on tags

### 3.4 Animations

| Element | Animation |
|---------|-----------|
| Page load | Fade in + translateY (0.4s ease-out) |
| Cards | Staggered entrance (0.1s delay each) |
| Progress ring | Smooth stroke-dashoffset transition |
| Status badge | Pulse animation for active states |
| File drop | Scale + glow on drag-over |
| Buttons | Scale 0.98 on active, shadow on hover |
| Mode switch | Slide indicator, fade content |

## 4. Technical Implementation

### 4.1 File Structure

```
frontend/src/
├── components/
│   ├── StatsCards.vue      # Statistics display
│   ├── UploadCard.vue      # File upload + config
│   ├── StatusCard.vue      # Task status + progress
│   └── RulesCard.vue       # Desensitization rules
├── composables/
│   └── useAnimation.js     # Animation utilities
├── App.vue                 # Main app (simplified)
└── style.css               # Global styles
```

### 4.2 Dependencies

- Vue 3 (existing)
- No new dependencies required
- Pure CSS animations

## 5. Acceptance Criteria

1. ✅ All 5 functional modes work identically
2. ✅ Page load animations smooth (60fps)
3. ✅ Progress ring animates smoothly
4. ✅ Drag & drop visual feedback works
5. ✅ Mobile responsive (breakpoints: 640px, 900px)
6. ✅ No new console errors
7. ✅ Build succeeds without warnings
