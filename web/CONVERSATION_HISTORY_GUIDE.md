# Conversation History Feature Guide

## Overview

The conversation history feature allows users to save, manage, and retrieve their chat conversations with the mental health assistant. All conversations are automatically saved to Firebase and can be accessed from any device.

## Features

### 🔄 Automatic Saving
- Conversations are automatically saved when you send messages
- No manual intervention required
- Real-time synchronization with Firebase

### 📱 Conversation Management
- **View History**: Access all your past conversations
- **Search**: Find specific conversations by content or title
- **Edit Titles**: Customize conversation titles for better organization
- **Archive/Unarchive**: Hide conversations without deleting them
- **Delete**: Permanently remove conversations
- **Continue**: Resume any previous conversation

### 📊 Statistics Dashboard
- Total conversations count
- Total messages count
- Active vs archived conversations
- Tone usage statistics

## How to Use

### Starting a New Conversation
1. Go to the chat page (`/chat`)
2. Choose your preferred conversation style
3. Start typing - your conversation will be automatically saved

### Accessing Conversation History
1. **From Chat Page**: Click the "History" button in the header
2. **From Home Page**: Click "View Conversation History" (if authenticated)
3. **Direct URL**: Navigate to `/conversation-history`

### Managing Conversations

#### Search Conversations
- Use the search bar to find conversations by content or title
- Search is performed in real-time as you type
- Clear search to return to full conversation list

#### Filter Conversations
- **Tone Filter**: Filter by conversation style (Professional, Caring, Balanced)
- **Archive Toggle**: Show/hide archived conversations

#### Edit Conversation Title
1. Click "Edit Title" on any conversation
2. Type the new title
3. Click "Save" or "Cancel"

#### Archive/Unarchive
- Click "Archive" to hide a conversation
- Click "Unarchive" to make it visible again
- Archived conversations are hidden by default

#### Delete Conversation
1. Click "Delete" on any conversation
2. Confirm the deletion in the popup
3. **Warning**: This action cannot be undone

#### Continue Conversation
- Click "Continue" to resume any previous conversation
- The conversation will load with all previous messages
- You can continue chatting from where you left off

### Navigation
- **Home Button**: Returns to the main page
- **New Conversation**: Starts a fresh conversation
- **Back to Chat**: Returns to the current chat session

## Technical Details

### Data Storage
- Conversations are stored in Firebase Firestore
- Each conversation includes:
  - Unique ID
  - Title
  - Messages array
  - Creation and update timestamps
  - Tone/style setting
  - Archive status
  - User association

### Security
- Users can only access their own conversations
- Data is protected by Firebase security rules
- Authentication required for all operations

### Performance
- Pagination support for large conversation lists
- Lazy loading for better performance
- Search optimization for quick results

## Troubleshooting

### Common Issues

**Conversation not saving**
- Check your internet connection
- Ensure you're logged in
- Try refreshing the page

**Can't find a conversation**
- Check if it's archived
- Try different search terms
- Verify you're using the correct account

**Slow loading**
- Conversations are loaded in batches
- Use search to find specific conversations faster
- Check your internet connection

### Support
If you encounter any issues:
1. Check the browser console for error messages
2. Try refreshing the page
3. Contact support if problems persist

## Privacy & Data

### Data Retention
- Conversations are stored indefinitely unless deleted
- You can delete conversations at any time
- Archived conversations are hidden but not deleted

### Data Export
- Currently, conversations are only accessible through the web interface
- Future updates may include export functionality

### Backup
- All data is automatically backed up by Firebase
- No manual backup required
- Data is replicated across multiple servers for reliability

## Future Enhancements

Planned features for future updates:
- Export conversations to PDF/JSON
- Conversation sharing (with privacy controls)
- Advanced search filters
- Conversation templates
- Bulk operations (delete multiple, archive multiple)
- Conversation analytics and insights 