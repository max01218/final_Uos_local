# Firestore Security Rules Setup

## 问题诊断

你遇到的 "Missing or insufficient permissions" 错误是因为Firestore安全规则没有正确配置。

## 解决方案

### 步骤1：访问Firebase Console

1. 打开 [Firebase Console](https://console.firebase.google.com/)
2. 选择你的项目 `uosfinal`
3. 在左侧菜单中点击 **Firestore Database**
4. 点击 **Rules** 标签页

### 步骤2：设置安全规则

将以下规则复制到Firestore Rules编辑器中：

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
    
    // Allow creation of new conversations
    match /conversations/{conversationId} {
      allow create: if request.auth != null && 
        request.auth.uid == request.resource.data.userId;
    }
  }
}
```

### 步骤3：发布规则

1. 点击 **Publish** 按钮
2. 等待规则生效（通常几秒钟）

### 步骤4：验证设置

规则生效后，你的应用应该能够：
- ✅ 创建新用户文档
- ✅ 保存对话记录
- ✅ 读取用户自己的对话
- ✅ 更新和删除对话

## 规则说明

- **用户文档**：用户只能访问自己的用户资料
- **对话文档**：用户只能访问自己创建的对话
- **创建权限**：允许已认证用户创建新对话
- **读写权限**：基于用户ID进行权限控制

## 测试步骤

1. 重启开发服务器：`npm run dev`
2. 重新注册/登录用户
3. 开始新对话
4. 检查History页面是否显示对话记录

## 常见问题

### Q: 规则发布后仍然有权限错误？
A: 清除浏览器缓存并重新登录

### Q: 如何测试规则是否生效？
A: 在Firebase Console > Firestore Database > Data 中查看是否有数据创建

### Q: 规则语法错误怎么办？
A: Firebase Console会显示具体的语法错误位置 