import React, { useState } from 'react';
import { Message, MessageType } from '@/types';
import { cn, formatTime } from '@/lib/utils';
import { 
  Copy, 
  ThumbsUp, 
  ThumbsDown, 
  AlertTriangle, 
  Lightbulb, 
  Clock,
  ChevronDown,
  ChevronUp
} from 'lucide-react';

interface MessageBubbleProps {
  message: Message;
  onFeedback?: (messageId: string, type: 'positive' | 'negative') => void;
  onCopy?: (content: string) => void;
  className?: string;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  onFeedback,
  onCopy,
  className
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);

  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';
  const isSystem = message.role === 'system';

  const getMessageClasses = (): string => {
    const baseClasses = 'max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-soft relative';
    
    if (isUser) {
      return cn(baseClasses, 'bg-gradient-to-br from-primary-500 to-primary-600 text-white ml-auto');
    }
    
    if (isSystem) {
      return cn(baseClasses, 'bg-secondary-100 text-secondary-700');
    }
    
    // Assistant message with different types
    switch (message.type) {
      case 'safety_alert':
        return cn(baseClasses, 'bg-gradient-to-br from-error-500 to-error-600 text-white border-2 border-error-400');
      case 'suggestion':
        return cn(baseClasses, 'bg-gradient-to-br from-mental-500 to-mental-600 text-white');
      case 'follow_up':
        return cn(baseClasses, 'bg-gradient-to-br from-warning-500 to-warning-600 text-white');
      default:
        return cn(baseClasses, 'bg-white text-secondary-900 border border-secondary-200');
    }
  };

  const getMessageIcon = () => {
    if (isUser) return null;
    
    switch (message.type) {
      case 'safety_alert':
        return <AlertTriangle className="h-4 w-4" />;
      case 'suggestion':
        return <Lightbulb className="h-4 w-4" />;
      case 'follow_up':
        return <Clock className="h-4 w-4" />;
      default:
        return null;
    }
  };

  const renderConfidenceIndicator = () => {
    if (!message.metadata?.confidence) return null;
    
    const confidence = message.metadata.confidence;
    const getConfidenceColor = (conf: number) => {
      if (conf >= 0.8) return 'text-success-500';
      if (conf >= 0.6) return 'text-warning-500';
      return 'text-error-500';
    };
    
    return (
      <div className="flex items-center gap-1 text-xs opacity-70">
        <div className={cn('w-2 h-2 rounded-full', getConfidenceColor(confidence))} />
        <span>{Math.round(confidence * 100)}%</span>
      </div>
    );
  };

  const renderSafetyNotes = () => {
    if (!message.metadata?.safety_notes?.length) return null;
    
    return (
      <div className="mt-2 p-2 bg-error-50 border border-error-200 rounded-lg">
        <div className="flex items-center gap-2 text-error-700 text-xs font-medium">
          <AlertTriangle className="h-3 w-3" />
          Safety Alert
        </div>
        <ul className="mt-1 text-xs text-error-600 space-y-1">
          {message.metadata.safety_notes.map((note, index) => (
            <li key={index}>• {note}</li>
          ))}
        </ul>
      </div>
    );
  };

  const renderFollowUpSuggestions = () => {
    if (!message.metadata?.follow_up_suggestions?.length) return null;
    
    return (
      <div className="mt-2 space-y-1">
        {message.metadata.follow_up_suggestions.map((suggestion, index) => (
          <button
            key={index}
            className="block w-full text-left p-2 text-xs bg-mental-50 hover:bg-mental-100 border border-mental-200 rounded-lg transition-colors"
            onClick={() => {
              // Handle suggestion click
              console.log('Suggestion clicked:', suggestion);
            }}
          >
            💡 {suggestion}
          </button>
        ))}
      </div>
    );
  };

  const renderMetadata = () => {
    if (!message.metadata) return null;
    
    const hasMetadata = message.metadata.confidence || 
                       message.metadata.processing_time ||
                       message.metadata.fusion_strategy;
    
    if (!hasMetadata) return null;
    
    return (
      <div className="mt-2 text-xs text-secondary-500 space-y-1">
        {message.metadata.processing_time && (
          <div>Processing time: {message.metadata.processing_time}ms</div>
        )}
        {message.metadata.fusion_strategy && (
          <div>Strategy: {message.metadata.fusion_strategy}</div>
        )}
      </div>
    );
  };

  return (
    <div className={cn('flex', isUser ? 'justify-end' : 'justify-start', className)}>
      <div className="relative group">
        {/* Message Content */}
        <div className={getMessageClasses()}>
          {/* Icon for special message types */}
          {getMessageIcon() && (
            <div className="absolute -top-2 -left-2 bg-white rounded-full p-1 shadow-soft">
              {getMessageIcon()}
            </div>
          )}
          
          {/* Message text */}
          <div className="text-balance">
            {message.content}
          </div>
          
          {/* Confidence indicator */}
          {renderConfidenceIndicator()}
          
          {/* Expandable metadata */}
          {isExpanded && (
            <>
              {renderSafetyNotes()}
              {renderFollowUpSuggestions()}
              {renderMetadata()}
            </>
          )}
        </div>
        
        {/* Message actions */}
        <div className="absolute -bottom-8 left-0 right-0 flex items-center justify-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          {isAssistant && (
            <>
              <button
                onClick={() => onFeedback?.(message.id, 'positive')}
                className="p-1 text-secondary-500 hover:text-success-500 transition-colors"
                aria-label="Positive feedback"
              >
                <ThumbsUp className="h-3 w-3" />
              </button>
              <button
                onClick={() => onFeedback?.(message.id, 'negative')}
                className="p-1 text-secondary-500 hover:text-error-500 transition-colors"
                aria-label="Negative feedback"
              >
                <ThumbsDown className="h-3 w-3" />
              </button>
            </>
          )}
          
          <button
            onClick={() => onCopy?.(message.content)}
            className="p-1 text-secondary-500 hover:text-primary-500 transition-colors"
            aria-label="Copy message"
          >
            <Copy className="h-3 w-3" />
          </button>
          
          {(message.metadata?.safety_notes?.length || 
            message.metadata?.follow_up_suggestions?.length ||
            message.metadata?.processing_time) && (
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="p-1 text-secondary-500 hover:text-primary-500 transition-colors"
              aria-label={isExpanded ? 'Collapse details' : 'Expand details'}
            >
              {isExpanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            </button>
          )}
        </div>
        
        {/* Timestamp */}
        <div className={cn(
          'text-xs text-secondary-400 mt-1',
          isUser ? 'text-right' : 'text-left'
        )}>
          {formatTime(message.timestamp)}
        </div>
      </div>
    </div>
  );
};

export default MessageBubble; 