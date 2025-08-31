import React, { useRef, useEffect, useCallback, useState, useMemo } from 'react';
import { Message } from '@/types';
import { cn, generateId, copyToClipboard } from '@/lib/utils';
import MessageBubble from './MessageBubble';
import { Loader2, RefreshCw } from 'lucide-react';
import Button from '@/components/ui/Button';

interface ChatWindowProps {
  messages: Message[];
  isLoading?: boolean;
  onLoadMore?: () => void;
  onFeedback?: (messageId: string, type: 'positive' | 'negative') => void;
  onExport?: () => void;
  onImport?: (messages: Message[]) => void;
  className?: string;
  maxHeight?: string;
  showScrollToBottom?: boolean;
}

const ChatWindow: React.FC<ChatWindowProps> = ({
  messages,
  isLoading = false,
  onLoadMore,
  onFeedback,
  onExport,
  onImport,
  className,
  maxHeight = 'calc(100vh - 200px)',
  showScrollToBottom = true
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [isScrolling, setIsScrolling] = useState(false);
  const [hasNewMessages, setHasNewMessages] = useState(false);

  // Performance optimization: use useMemo to cache message list
  const visibleMessages = useMemo(() => {
    return messages.slice(-50); // Only show the last 50 messages for better performance
  }, [messages]);

  // Auto scroll to bottom
  const scrollToBottom = useCallback((behavior: ScrollBehavior = 'smooth') => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior });
    }
  }, []);

  // Check if scroll button should be shown
  const checkScrollPosition = useCallback(() => {
    if (!containerRef.current) return;
    
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
    setShowScrollButton(!isNearBottom);
  }, []);

  // Handle scroll events
  const handleScroll = useCallback(() => {
    if (!isScrolling) {
      setIsScrolling(true);
      setTimeout(() => setIsScrolling(false), 100);
    }
    checkScrollPosition();
  }, [isScrolling, checkScrollPosition]);

  // Listen for new messages
  useEffect(() => {
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      
      // Ensure timestamp is a Date object
      const timestamp = lastMessage.timestamp instanceof Date 
        ? lastMessage.timestamp 
        : new Date(lastMessage.timestamp);
      
      const isRecent = Date.now() - timestamp.getTime() < 5000;
      
      if (isRecent && !isScrolling) {
        setHasNewMessages(true);
        scrollToBottom();
        setTimeout(() => setHasNewMessages(false), 3000);
      }
    }
  }, [messages, isScrolling, scrollToBottom]);

  // Initial scroll to bottom
  useEffect(() => {
    scrollToBottom('auto');
  }, [scrollToBottom]);

  // Handle feedback
  const handleFeedback = useCallback((messageId: string, type: 'positive' | 'negative') => {
    onFeedback?.(messageId, type);
  }, [onFeedback]);

  // Handle copy message
  const handleCopyMessage = useCallback(async (content: string) => {
    const success = await copyToClipboard(content);
    if (success) {
      // Can add toast notification
      console.log('Message copied to clipboard');
    }
  }, []);



  // Render loading indicator
  const renderLoadingIndicator = () => (
    <div className="flex justify-start mb-4">
        <div className="flex items-center gap-2 p-3 bg-gray-100 rounded-2xl">
            <div className="text-sm text-gray-500 animate-pulse">
                assistant is typing…
            </div>
        </div>
    </div>
  );

  // Render empty state
  const renderEmptyState = () => (
    <div className="flex flex-col items-center justify-center h-64 text-center">
      <div className="w-16 h-16 bg-secondary-100 rounded-full flex items-center justify-center mb-4">
        <svg className="w-8 h-8 text-secondary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
        </svg>
      </div>
      <h3 className="text-lg font-medium text-secondary-900 mb-2">Start a Conversation</h3>
      <p className="text-secondary-600 max-w-md">
        Welcome to the Mental Health Assistant. Please share your thoughts, and I'll do my best to provide support and help.
      </p>
    </div>
  );

  // Render toolbar
  const renderToolbar = () => (
    <div className="flex items-center justify-between p-4 border-b border-secondary-200 bg-white rounded-t-xl">
      <div className="flex items-center gap-2">
        <h3 className="text-sm font-medium text-secondary-900">Conversation History</h3>
        <span className="text-xs text-secondary-500">({messages.length} messages)</span>
      </div>
      
      <div className="flex items-center gap-2">
        {onLoadMore && messages.length > 50 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onLoadMore}
            className="text-xs"
          >
            <RefreshCw className="h-3 w-3 mr-1" />
            Load More
          </Button>
        )}
        

      </div>
    </div>
  );

  return (
    <div className={cn('flex flex-col bg-white rounded-xl shadow-soft overflow-hidden', className)}>
      {/* Toolbar */}
      {renderToolbar()}
      
      {/* Message container */}
      <div
        ref={containerRef}
        className="flex-1 overflow-y-auto scrollbar-thin"
        style={{ maxHeight }}
        onScroll={handleScroll}
      >
        <div className="p-4 space-y-4">
          {/* Empty state */}
          {messages.length === 0 && !isLoading && renderEmptyState()}
          
          {/* Message list */}
          {visibleMessages.map((message, index) => (
            <div key={message.id}>
              <MessageBubble
                message={message}
                onFeedback={handleFeedback}
                onCopy={handleCopyMessage}
              />
            </div>
          ))}
          
          {/* Loading indicator */}
          {isLoading && renderLoadingIndicator()}
          
          {/* Scroll to bottom anchor */}
          <div ref={messagesEndRef} />
        </div>
      </div>
      
      {/* Scroll to bottom button */}
      {showScrollButton && showScrollToBottom && (
        <button
          className="absolute bottom-4 right-4 p-3 bg-white border border-secondary-200 rounded-full shadow-medium hover:shadow-strong transition-all duration-200"
          onClick={() => scrollToBottom()}
          aria-label="Scroll to bottom"
        >
          <svg className="w-4 h-4 text-secondary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          </svg>
        </button>
      )}
      
      {/* New message indicator */}
      {hasNewMessages && (
        <div className="absolute bottom-16 right-4 px-3 py-1 bg-primary-500 text-white text-xs rounded-full">
          New message
        </div>
      )}
    </div>
  );
};

export default ChatWindow; 