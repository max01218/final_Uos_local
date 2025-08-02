# 对话保存功能测试指南

## 测试目标
验证用户和助手的消息都能正确保存到Firebase中。

## 测试步骤

### 1. 启动服务
```bash
cd web
npm run dev
```

### 2. 登录测试
1. 访问 http://localhost:3000
2. 点击 "Sign In" 或 "Get Started"
3. 使用现有账户登录或创建新账户

### 3. 开始对话测试
1. 点击 "Start Chat" 或选择语气
2. 发送一条用户消息，例如：
   ```
   "What relaxation techniques can I try?"
   ```
3. 等待助手回复
4. 再发送一条用户消息，例如：
   ```
   "Can you explain more about deep breathing?"
   ```
5. 等待助手回复

### 4. 检查保存结果

#### 方法1：检查History页面
1. 点击页面右上角的 "History" 按钮
2. 查看对话列表
3. 确认：
   - 对话标题正确显示
   - 消息数量显示正确（应该是4条：2条用户消息 + 2条助手消息）
   - 最后更新时间正确

#### 方法2：检查Firebase Console
1. 访问 [Firebase Console](https://console.firebase.google.com/)
2. 选择你的项目
3. 点击 "Firestore Database"
4. 查看 `conversations` 集合
5. 点击最新的对话文档
6. 展开 `messages` 字段
7. 确认：
   - 包含用户消息（role: "user"）
   - 包含助手消息（role: "assistant"）
   - 消息按时间顺序排列
   - 时间戳正确

#### 方法3：检查浏览器控制台
1. 按 F12 打开开发者工具
2. 点击 "Console" 标签页
3. 发送消息时观察控制台输出
4. 确认没有错误信息

### 5. 验证数据完整性

#### 检查消息结构
每个消息应该包含：
```json
{
  "id": "unique_id",
  "role": "user" | "assistant",
  "content": "消息内容",
  "timestamp": "时间戳",
  "metadata": {
    // 助手消息的额外信息
  }
}
```

#### 检查对话结构
每个对话应该包含：
```json
{
  "id": "conv_timestamp_random",
  "title": "对话标题",
  "messages": [
    // 用户和助手的消息数组
  ],
  "userId": "用户ID",
  "createdAt": "创建时间",
  "updatedAt": "更新时间",
  "tone": "professional"
}
```

## 预期结果

### ✅ 成功指标
1. **用户消息保存**：每条用户消息都保存到Firebase
2. **助手消息保存**：每条助手消息都保存到Firebase
3. **消息顺序**：消息按发送顺序正确排列
4. **时间戳**：所有消息都有正确的时间戳
5. **数据一致性**：本地状态和Firebase数据一致
6. **错误处理**：没有保存相关的错误信息

### ❌ 失败指标
1. **消息丢失**：某些消息没有保存
2. **顺序错误**：消息顺序不正确
3. **时间戳错误**：时间戳格式或值不正确
4. **类型错误**：消息类型不正确
5. **权限错误**：Firebase权限问题

## 故障排除

### 如果只有一边的消息被保存
1. 检查 `chat.tsx` 中的 `handleSend` 函数
2. 确认用户消息和助手消息都调用了 `addMessage`
3. 检查 `useConversations.ts` 中的 `addMessage` 函数
4. 检查 `firebase.ts` 中的 `addMessageToConversation` 函数

### 如果消息没有保存
1. 检查Firebase配置是否正确
2. 检查用户是否已登录
3. 检查Firestore安全规则
4. 检查网络连接

### 如果出现错误
1. 查看浏览器控制台错误信息
2. 查看Firebase Console错误日志
3. 检查TypeScript编译错误

## 测试完成后的验证

完成测试后，你应该能够：
1. 在History页面看到完整的对话
2. 在Firebase Console中看到正确的数据结构
3. 确认用户和助手的消息都被正确保存
4. 验证消息的时间戳和顺序正确 