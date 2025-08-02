# 心理健康助手 - 现代化前端

这是一个基于 Next.js 和 TypeScript 构建的现代化心理健康助手前端应用。

## ✨ 特性

### 🎨 设计系统
- **统一的设计语言**: 使用 Tailwind CSS 构建的一致设计系统
- **响应式设计**: 完美适配桌面端和移动端
- **可访问性**: 支持键盘导航和屏幕阅读器
- **动画效果**: 使用 Framer Motion 提供流畅的交互体验

### 🧩 组件化架构
- **模块化组件**: 可复用的 UI 组件库
- **类型安全**: 完整的 TypeScript 类型定义
- **性能优化**: 使用 React.memo 和 useMemo 优化渲染性能
- **状态管理**: 清晰的状态管理架构

### 🚀 现代化工具
- **Next.js 14**: 最新的 React 框架
- **Tailwind CSS**: 实用优先的 CSS 框架
- **Framer Motion**: 强大的动画库
- **Lucide React**: 现代化的图标库
- **React Hook Form**: 高性能的表单处理

### 💡 智能功能
- **情绪检测**: 实时分析用户情绪状态
- **安全提醒**: 自动检测潜在风险并提供帮助
- **智能建议**: 基于上下文的对话建议
- **反馈系统**: 用户反馈收集和分析

## 🛠️ 技术栈

- **框架**: Next.js 14
- **语言**: TypeScript
- **样式**: Tailwind CSS
- **动画**: Framer Motion
- **图标**: Lucide React
- **表单**: React Hook Form
- **工具**: clsx, tailwind-merge
- **通知**: React Hot Toast

## 📦 安装和运行

### 前置要求
- Node.js 18+ 
- npm 或 yarn

### 安装依赖
```bash
cd web
npm install
```

### 开发模式
```bash
npm run dev
```

### 构建生产版本
```bash
npm run build
npm start
```

### 代码检查
```bash
npm run lint
npm run type-check
```

## 🏗️ 项目结构

```
web/
├── src/
│   ├── components/
│   │   ├── ui/              # 基础UI组件
│   │   │   ├── Button.tsx
│   │   │   └── Input.tsx
│   │   └── chat/            # 聊天相关组件
│   │       ├── ChatWindow.tsx
│   │       ├── ChatInput.tsx
│   │       └── MessageBubble.tsx
│   ├── lib/                 # 工具函数
│   │   └── utils.ts
│   ├── types/               # TypeScript类型定义
│   │   └── index.ts
│   ├── styles/              # 全局样式
│   │   └── globals.css
│   └── pages/               # 页面组件
│       ├── _app.tsx
│       └── chat.tsx
├── public/                  # 静态资源
├── tailwind.config.js       # Tailwind配置
├── tsconfig.json           # TypeScript配置
├── next.config.js          # Next.js配置
└── package.json
```

## 🎨 设计系统

### 颜色系统
- **主色调**: 蓝色系 (primary)
- **辅助色**: 灰色系 (secondary)
- **功能色**: 成功绿、警告橙、错误红
- **主题色**: 心理健康紫色 (mental)

### 组件规范
- **按钮**: 支持多种变体和尺寸
- **输入框**: 带验证和状态指示
- **卡片**: 统一的阴影和圆角
- **消息气泡**: 不同类型消息的视觉区分

### 响应式断点
- **移动端**: < 768px
- **平板端**: 768px - 1024px
- **桌面端**: > 1024px

## 🔧 开发指南

### 添加新组件
1. 在 `src/components/` 下创建组件文件
2. 使用 TypeScript 定义 Props 接口
3. 添加适当的 ARIA 标签
4. 使用 Tailwind CSS 类名
5. 添加动画效果（可选）

### 样式指南
- 优先使用 Tailwind CSS 类名
- 自定义样式使用 `@layer` 指令
- 保持颜色和间距的一致性
- 考虑深色模式支持

### 性能优化
- 使用 `React.memo` 包装纯组件
- 使用 `useMemo` 缓存计算结果
- 使用 `useCallback` 缓存函数
- 实现虚拟滚动处理大量数据

## 🧪 测试

### 单元测试
```bash
npm test
```

### E2E测试
```bash
npm run test:e2e
```

## 📱 移动端适配

- 响应式布局设计
- 触摸友好的交互
- 移动端优化的输入体验
- 适配不同屏幕尺寸

## ♿ 可访问性

- 语义化 HTML 结构
- ARIA 标签支持
- 键盘导航支持
- 屏幕阅读器兼容
- 颜色对比度符合 WCAG 标准

## 🚀 部署

### Vercel 部署
```bash
npm run build
vercel --prod
```

### 其他平台
构建产物位于 `.next` 目录，可部署到任何支持 Node.js 的平台。

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 支持

如有问题，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至 [your-email@example.com]

---

**注意**: 这是一个心理健康相关的应用，请确保在使用时遵循相关的隐私和安全规范。 