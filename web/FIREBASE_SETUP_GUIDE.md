# Firebase 设置指南

本指南将帮助你设置 Firebase 来存储用户数据和对话记录。

## 1. 创建 Firebase 项目

### 步骤 1: 访问 Firebase Console
1. 前往 [Firebase Console](https://console.firebase.google.com/)
2. 点击 "创建项目" 或 "Add project"
3. 输入项目名称（例如：`icd11-mental-health-assistant`）
4. 选择是否启用 Google Analytics（可选）
5. 点击 "创建项目"

### 步骤 2: 启用 Authentication
1. 在项目控制台中，点击左侧菜单的 "Authentication"
2. 点击 "开始使用" 或 "Get started"
3. 在 "Sign-in method" 标签页中，启用 "Email/Password"
4. 点击 "保存"

### 步骤 3: 创建 Firestore 数据库
1. 在左侧菜单中点击 "Firestore Database"
2. 点击 "创建数据库"
3. 选择 "以测试模式开始"（稍后可以设置安全规则）
4. 选择数据库位置（建议选择离用户最近的区域）
5. 点击 "完成"

## 2. 获取 Firebase 配置

### 步骤 1: 添加 Web 应用
1. 在项目控制台中，点击齿轮图标（⚙️）选择 "项目设置"
2. 在 "常规" 标签页中，滚动到 "您的应用" 部分
3. 点击 Web 图标（</>）
4. 输入应用昵称（例如：`ICD-11 Web App`）
5. 点击 "注册应用"

### 步骤 2: 复制配置
1. 复制显示的 Firebase 配置对象
2. 创建 `.env.local` 文件（在 `web` 目录下）
3. 将配置信息添加到环境变量中

## 3. 配置环境变量

在 `web` 目录下创建 `.env.local` 文件：

```bash
# Firebase Configuration
NEXT_PUBLIC_FIREBASE_API_KEY=your_actual_api_key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your_project_id.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your_project_id
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your_project_id.appspot.com
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your_messaging_sender_id
NEXT_PUBLIC_FIREBASE_APP_ID=your_app_id
```

## 4. 设置 Firestore 安全规则

在 Firestore Database 中，点击 "规则" 标签页，使用以下安全规则：

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Users can only access their own data
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
    
    // Users can only access their own conversations
    match /conversations/{conversationId} {
      allow read, write: if request.auth != null && 
        request.auth.uid == resource.data.userId;
    }
    
    // Deny all other access
    match /{document=**} {
      allow read, write: if false;
    }
  }
}
```

## 5. 数据库结构

Firebase 将自动创建以下集合：

### users 集合
```javascript
{
  id: "user_uid",
  email: "user@example.com",
  name: "User Name",
  createdAt: Timestamp,
  lastActive: Timestamp,
  isVerified: boolean,
  preferences: {
    theme: "light",
    language: "en",
    notifications: true
  },
  updatedAt: Timestamp
}
```

### conversations 集合
```javascript
{
  id: "conversation_id",
  userId: "user_uid",
  title: "Conversation Title",
  messages: [
    {
      id: "message_id",
      role: "user|assistant|system",
      content: "Message content",
      timestamp: Timestamp,
      type: "normal|safety_alert|suggestion|follow_up",
      metadata: {
        confidence: 0.95,
        processing_time: 1200,
        fusion_strategy: "weighted",
        safety_notes: ["Note 1", "Note 2"],
        follow_up_suggestions: ["Suggestion 1", "Suggestion 2"]
      }
    }
  ],
  createdAt: Timestamp,
  updatedAt: Timestamp
}
```

## 6. 安装依赖

运行以下命令安装 Firebase 依赖：

```bash
cd web
npm install firebase
```

## 7. 测试设置

1. 启动开发服务器：
   ```bash
   npm run dev
   ```

2. 访问应用并尝试注册/登录
3. 检查 Firebase Console 中的 Authentication 和 Firestore 部分，确认数据已正确保存

## 8. 生产环境注意事项

### 安全规则
- 在生产环境中，确保 Firestore 安全规则足够严格
- 考虑添加额外的验证逻辑

### 环境变量
- 确保生产环境的环境变量正确设置
- 不要将 `.env.local` 文件提交到版本控制系统

### 监控
- 在 Firebase Console 中设置监控和警报
- 定期检查使用量和性能指标

## 9. 故障排除

### 常见问题

1. **配置错误**
   - 确保所有环境变量都正确设置
   - 检查 Firebase 项目 ID 是否匹配

2. **权限错误**
   - 确保 Firestore 安全规则允许用户访问自己的数据
   - 检查 Authentication 是否已启用

3. **网络错误**
   - 检查网络连接
   - 确保 Firebase 项目在正确的区域

### 调试技巧

1. 在浏览器控制台中检查 Firebase 初始化
2. 使用 Firebase Console 查看实时数据
3. 检查网络请求中的错误信息

## 10. 下一步

设置完成后，你可以：

1. 自定义用户界面和用户体验
2. 添加更多功能，如密码重置、邮箱验证等
3. 实现更复杂的查询和过滤功能
4. 添加数据分析和报告功能

如有问题，请参考 [Firebase 官方文档](https://firebase.google.com/docs) 或联系技术支持。 