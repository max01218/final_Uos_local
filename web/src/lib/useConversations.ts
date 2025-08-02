import { useState, useEffect, useCallback } from 'react';
import { Conversation, Message } from '@/types';
import { 
  saveConversation, 
  getUserConversations, 
  updateConversation 
} from './firebase';
import { useAuth } from './AuthContext';

export function useConversations() {
  const { user } = useAuth();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load user conversations
  const loadConversations = useCallback(async () => {
    if (!user) return;

    try {
      setLoading(true);
      setError(null);
      const userConversations = await getUserConversations(user.id);
      setConversations(userConversations);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load conversations';
      setError(errorMessage);
      console.error('Error loading conversations:', err);
    } finally {
      setLoading(false);
    }
  }, [user]);

  // Create new conversation
  const createConversation = useCallback(async (title: string, initialMessage?: Message): Promise<Conversation> => {
    if (!user) {
      throw new Error('User not authenticated');
    }

    const newConversation: Conversation = {
      id: `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      title,
      messages: initialMessage ? [initialMessage] : [],
      userId: user.id,
      createdAt: new Date(),
      updatedAt: new Date(),
      tone: 'professional'
    };

    try {
      await saveConversation(user.id, newConversation);
      setConversations(prev => [newConversation, ...prev]);
      return newConversation;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to create conversation';
      setError(errorMessage);
      throw err;
    }
  }, [user]);

  // Add message to conversation
  const addMessageToConversation = useCallback(async (
    conversationId: string, 
    message: Message
  ) => {
    if (!user) {
      throw new Error('User not authenticated');
    }

    try {
      const conversation = conversations.find(c => c.id === conversationId);
      if (!conversation) {
        throw new Error('Conversation not found');
      }

      const updatedConversation: Conversation = {
        ...conversation,
        messages: [...conversation.messages, message],
        updatedAt: new Date()
      };

      await updateConversation(conversationId, updatedConversation);
      
      setConversations(prev => 
        prev.map(c => 
          c.id === conversationId 
            ? updatedConversation 
            : c
        )
      );

      return updatedConversation;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to add message';
      setError(errorMessage);
      throw err;
    }
  }, [user, conversations]);

  // Update conversation title
  const updateConversationTitle = useCallback(async (
    conversationId: string, 
    newTitle: string
  ) => {
    if (!user) {
      throw new Error('User not authenticated');
    }

    try {
      await updateConversation(conversationId, { 
        title: newTitle,
        updatedAt: new Date()
      });
      
      setConversations(prev => 
        prev.map(c => 
          c.id === conversationId 
            ? { ...c, title: newTitle, updatedAt: new Date() }
            : c
        )
      );
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to update conversation title';
      setError(errorMessage);
      throw err;
    }
  }, [user]);

  // Delete conversation
  const deleteConversation = useCallback(async (conversationId: string) => {
    if (!user) {
      throw new Error('User not authenticated');
    }

    try {
      // Note: You'll need to implement deleteConversation in firebase.ts
      // For now, we'll just remove it from local state
      setConversations(prev => prev.filter(c => c.id !== conversationId));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete conversation';
      setError(errorMessage);
      throw err;
    }
  }, [user]);

  // Get conversation by ID
  const getConversation = useCallback((conversationId: string): Conversation | undefined => {
    return conversations.find(c => c.id === conversationId);
  }, [conversations]);

  // Load conversations when user changes
  useEffect(() => {
    if (user) {
      loadConversations();
    } else {
      setConversations([]);
    }
  }, [user, loadConversations]);

  return {
    conversations,
    loading,
    error,
    loadConversations,
    createConversation,
    addMessageToConversation,
    updateConversationTitle,
    deleteConversation,
    getConversation,
    clearError: () => setError(null)
  };
} 