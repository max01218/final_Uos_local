import { useState, useEffect, useCallback } from 'react';
import { useAuth } from './AuthContext';
import { 
  saveConversation, 
  getUserConversations, 
  updateConversation,
  deleteConversation,
  getUserConversationsPaginated,
  searchConversations,
  getConversationStats,
  addMessageToConversation
} from './firebase';
import { Conversation, Message, ToneType } from '@/types';

interface ConversationStats {
  totalConversations: number;
  totalMessages: number;
  toneStats: Record<string, number>;
}

interface UseConversationsReturn {
  conversations: Conversation[];
  currentConversation: Conversation | null;
  loading: boolean;
  error: string | null;
  stats: ConversationStats | null;
  hasMore: boolean;
  lastDoc: any;
  
  // Basic operations
  createConversation: (title: string, tone?: ToneType) => Promise<Conversation>;
  addMessage: (conversationId: string, message: Message) => Promise<void>;
  updateConversationTitle: (conversationId: string, title: string) => Promise<void>;
  deleteConversationById: (conversationId: string) => Promise<void>;
  
  // Advanced operations
  loadConversations: (pageSize?: number) => Promise<void>;
  loadMoreConversations: () => Promise<void>;
  searchConversationsByTerm: (searchTerm: string) => Promise<void>;
  loadConversationStats: () => Promise<void>;
  
  // State management
  setCurrentConversation: (conversation: Conversation | null) => void;
  clearError: () => void;
}

export const useConversations = (): UseConversationsReturn => {
  const { user } = useAuth();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversation, setCurrentConversation] = useState<Conversation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<ConversationStats | null>(null);
  const [hasMore, setHasMore] = useState(true);
  const [lastDoc, setLastDoc] = useState<any>(null);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const createConversation = useCallback(async (title: string, tone: ToneType = 'professional'): Promise<Conversation> => {
    if (!user) throw new Error('User not authenticated');

    const newConversation: Conversation = {
      id: `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      title,
      messages: [],
      userId: user.id,
      createdAt: new Date(),
      updatedAt: new Date(),
      tone: tone
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

  const addMessage = useCallback(async (conversationId: string, message: Message): Promise<void> => {
    if (!user) throw new Error('User not authenticated');

    try {
      // Use the dedicated function to add message to conversation
      await addMessageToConversation(conversationId, message);
      
      // Update local state
      const conversation = conversations.find(c => c.id === conversationId);
      if (conversation) {
        const updatedConversation: Conversation = {
          ...conversation,
          messages: [...conversation.messages, message],
          updatedAt: new Date()
        };
        
        setConversations(prev => 
          prev.map(c => c.id === conversationId ? updatedConversation : c)
        );

        if (currentConversation?.id === conversationId) {
          setCurrentConversation(updatedConversation);
        }
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to add message';
      setError(errorMessage);
      throw err;
    }
  }, [user, conversations, currentConversation]);

  const updateConversationTitle = useCallback(async (conversationId: string, title: string): Promise<void> => {
    if (!user) throw new Error('User not authenticated');

    try {
      await updateConversation(conversationId, { title });
      
      setConversations(prev => 
        prev.map(c => c.id === conversationId ? { ...c, title, updatedAt: new Date() } : c)
      );

      if (currentConversation?.id === conversationId) {
        setCurrentConversation(prev => prev ? { ...prev, title, updatedAt: new Date() } : null);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to update conversation title';
      setError(errorMessage);
      throw err;
    }
  }, [user, currentConversation]);

  const deleteConversationById = useCallback(async (conversationId: string): Promise<void> => {
    if (!user) throw new Error('User not authenticated');

    try {
      await deleteConversation(conversationId);
      
      setConversations(prev => prev.filter(c => c.id !== conversationId));
      
      if (currentConversation?.id === conversationId) {
        setCurrentConversation(null);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete conversation';
      setError(errorMessage);
      throw err;
    }
  }, [user, currentConversation]);

  const loadConversations = useCallback(async (pageSize: number = 10): Promise<void> => {
    if (!user) return;

    setLoading(true);
    setError(null);

    try {
      const result = await getUserConversationsPaginated(user.id, pageSize);
      setConversations(result.conversations);
      setLastDoc(result.lastDoc);
      setHasMore(result.hasMore);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load conversations';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [user]);

  const loadMoreConversations = useCallback(async (): Promise<void> => {
    if (!user || !hasMore || loading) return;

    setLoading(true);
    setError(null);

    try {
      const result = await getUserConversationsPaginated(user.id, 10, lastDoc);
      setConversations(prev => [...prev, ...result.conversations]);
      setLastDoc(result.lastDoc);
      setHasMore(result.hasMore);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load more conversations';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [user, hasMore, loading, lastDoc]);

  const searchConversationsByTerm = useCallback(async (searchTerm: string): Promise<void> => {
    if (!user || !searchTerm.trim()) {
      await loadConversations();
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const searchResults = await searchConversations(user.id, searchTerm);
      setConversations(searchResults);
      setHasMore(false);
      setLastDoc(null);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to search conversations';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [user, loadConversations]);

  const loadConversationStats = useCallback(async (): Promise<void> => {
    if (!user) return;

    try {
      const conversationStats = await getConversationStats(user.id);
      setStats(conversationStats);
    } catch (err) {
      console.error('Failed to load conversation stats:', err);
    }
  }, [user]);

  // Load conversations on mount
  useEffect(() => {
    if (user) {
      loadConversations();
      loadConversationStats();
    }
  }, [user, loadConversations, loadConversationStats]);

  return {
    conversations,
    currentConversation,
    loading,
    error,
    stats,
    hasMore,
    lastDoc,
    
    // Basic operations
    createConversation,
    addMessage,
    updateConversationTitle,
    deleteConversationById,
    
    // Advanced operations
    loadConversations,
    loadMoreConversations,
    searchConversationsByTerm,
    loadConversationStats,
    
    // State management
    setCurrentConversation,
    clearError
  };
}; 