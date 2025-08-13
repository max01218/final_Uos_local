import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '@/lib/AuthContext';
import { useConversations } from '@/lib/useConversations';
import HomeButton from '@/components/ui/HomeButton';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import { 
  Search, 
  MessageSquare, 
  Calendar, 
  Clock, 
  Trash2, 
  Edit3,
  MoreHorizontal,
  ChevronDown,
  Filter
} from 'lucide-react';

interface ConversationHistoryProps {}

const ConversationHistory: React.FC<ConversationHistoryProps> = () => {
  const router = useRouter();
  const { user } = useAuth();
  const {
    conversations,
    loading,
    error,
    stats,
    hasMore,
    searchConversationsByTerm,
    loadMoreConversations,
    loadConversations,
    deleteConversationById,
    updateConversationTitle,
    clearError
  } = useConversations();

  const [searchTerm, setSearchTerm] = useState('');
  const [editingTitle, setEditingTitle] = useState<string | null>(null);
  const [newTitle, setNewTitle] = useState('');
  const [filterTone, setFilterTone] = useState<string>('all');

  // Redirect if not authenticated
  useEffect(() => {
    if (!user) {
      router.push('/auth');
    }
  }, [user, router]);

  // Handle search with debounce
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (searchTerm.trim()) {
        searchConversationsByTerm(searchTerm);
      } else {
        // Reload conversations when search is cleared
        loadConversations();
      }
    }, 500);

    return () => clearTimeout(timeoutId);
  }, [searchTerm, searchConversationsByTerm, loadConversations]);

  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
  };

  const handleContinueConversation = (conversationId: string) => {
    router.push(`/chat?conversation=${conversationId}`);
  };

  const handleDeleteConversation = async (conversationId: string) => {
    if (window.confirm('Are you sure you want to delete this conversation? This action cannot be undone.')) {
      try {
        await deleteConversationById(conversationId);
      } catch (err) {
        console.error('Failed to delete conversation:', err);
      }
    }
  };

  const handleEditTitle = (conversationId: string, currentTitle: string) => {
    setEditingTitle(conversationId);
    setNewTitle(currentTitle);
  };

  const handleSaveTitle = async (conversationId: string) => {
    try {
      await updateConversationTitle(conversationId, newTitle);
      setEditingTitle(null);
      setNewTitle('');
    } catch (err) {
      console.error('Failed to update title:', err);
    }
  };

  const handleCancelEdit = () => {
    setEditingTitle(null);
    setNewTitle('');
  };

  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };

  const getToneLabel = (tone: string) => {
    const toneLabels: Record<string, string> = {
      'professional': 'Professional',
      'caring': 'Caring',
      'empathetic_professional': 'Balanced'
    };
    return toneLabels[tone] || tone;
  };

  const filteredConversations = conversations.filter(conversation => {
    if (filterTone !== 'all' && conversation.tone !== filterTone) {
      return false;
    }
    return true;
  });

  if (!user) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading user authentication...</p>
        </div>
      </div>
    );
  }

  // Debug information
  console.log('ConversationHistory Debug:', {
    user: user ? { id: user.id, email: user.email } : null,
    conversationsCount: conversations.length,
    loading,
    error,
    stats
  });

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <HomeButton variant="ghost" size="sm" />
              <h1 className="text-2xl font-bold text-gray-900">Conversation History</h1>
            </div>
            <div className="flex items-center space-x-4">
              <Button
                onClick={() => router.push('/chat')}
                variant="primary"
                size="sm"
              >
                New Conversation
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Cards */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <MessageSquare className="h-8 w-8 text-blue-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Total Conversations</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.totalConversations}</p>
                </div>
              </div>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <MessageSquare className="h-8 w-8 text-green-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Total Messages</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.totalMessages}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Search and Filters */}
        <div className="bg-white rounded-lg shadow mb-6">
          <div className="p-6">
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="flex-1">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <Input
                    type="text"
                    placeholder="Search conversations..."
                    value={searchTerm}
                    onChange={handleSearch}
                    className="pl-10"
                  />
                </div>
              </div>
              <div className="flex gap-2">
                <select
                  value={filterTone}
                  onChange={(e) => setFilterTone(e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="all">All Tones</option>
                  <option value="professional">Professional</option>
                  <option value="caring">Caring</option>
                  <option value="empathetic_professional">Balanced</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-6">
            <div className="flex">
              <div className="flex-shrink-0">
                <div className="text-red-400">Error: {error}</div>
              </div>
              <div className="ml-auto pl-3">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={clearError}
                >
                  Dismiss
                </Button>
              </div>
            </div>
          </div>
        )}

        {/* Conversations List */}
        <div className="bg-white rounded-lg shadow">
          {loading && conversations.length === 0 ? (
            <div className="p-8 text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
              <p className="mt-2 text-gray-600">Loading conversations...</p>
            </div>
          ) : filteredConversations.length === 0 ? (
            <div className="p-8 text-center">
              <MessageSquare className="h-12 w-12 text-gray-400 mx-auto" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">No conversations found</h3>
              <p className="mt-1 text-sm text-gray-500">
                {searchTerm ? 'Try adjusting your search terms.' : 'Start a new conversation to get started.'}
              </p>
            </div>
          ) : (
            <div className="divide-y divide-gray-200">
              {filteredConversations.map((conversation) => (
                <div key={conversation.id} className="p-6 hover:bg-gray-50 transition-colors">
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-3">
                        {editingTitle === conversation.id ? (
                          <div className="flex items-center space-x-2">
                            <Input
                              value={newTitle}
                                                             onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewTitle(e.target.value)}
                              className="flex-1"
                              autoFocus
                            />
                            <Button
                              size="sm"
                              onClick={() => handleSaveTitle(conversation.id)}
                            >
                              Save
                            </Button>
                            <Button
                              variant="secondary"
                              size="sm"
                              onClick={handleCancelEdit}
                            >
                              Cancel
                            </Button>
                          </div>
                        ) : (
                          <h3 className="text-lg font-medium text-gray-900 truncate">
                            {conversation.title}
                          </h3>
                        )}
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                          {getToneLabel(conversation.tone)}
                        </span>
                      </div>
                      
                      <div className="mt-2 flex items-center space-x-4 text-sm text-gray-500">
                        <div className="flex items-center">
                          <MessageSquare className="h-4 w-4 mr-1" />
                          {conversation.messages.length} messages
                        </div>
                        <div className="flex items-center">
                          <Calendar className="h-4 w-4 mr-1" />
                          {formatDate(conversation.createdAt)}
                        </div>
                        <div className="flex items-center">
                          <Clock className="h-4 w-4 mr-1" />
                          {formatDate(conversation.updatedAt)}
                        </div>
                      </div>

                      {conversation.messages.length > 0 && (
                        <p className="mt-2 text-sm text-gray-600 line-clamp-2">
                          {conversation.messages[conversation.messages.length - 1].content}
                        </p>
                      )}
                    </div>

                    <div className="flex items-center space-x-2 ml-4">
                      <Button
                        variant="primary"
                        size="sm"
                        onClick={() => handleContinueConversation(conversation.id)}
                      >
                        Continue
                      </Button>
                      
                      <div className="relative">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => {
                            // Toggle dropdown menu
                          }}
                        >
                          <MoreHorizontal className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </div>

                  {/* Action buttons */}
                  <div className="mt-4 flex items-center space-x-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleEditTitle(conversation.id, conversation.title)}
                    >
                      <Edit3 className="h-4 w-4 mr-1" />
                      Edit Title
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDeleteConversation(conversation.id)}
                      className="text-red-600 hover:text-red-700"
                    >
                      <Trash2 className="h-4 w-4 mr-1" />
                      Delete
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Load More Button */}
          {hasMore && !loading && (
            <div className="p-6 border-t border-gray-200">
              <Button
                onClick={loadMoreConversations}
                variant="secondary"
                className="w-full"
                disabled={loading}
              >
                {loading ? 'Loading...' : 'Load More Conversations'}
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ConversationHistory; 