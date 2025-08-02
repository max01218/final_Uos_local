import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Smile, AlertTriangle, Sparkles, Mic } from 'lucide-react';
import { cn, debounce } from '@/lib/utils';
import Button from '@/components/ui/Button';
import { EmotionAnalysis, SafetyAlert } from '@/types';

interface ChatInputProps {
  onSend: (message: string, metadata?: any) => void;
  disabled?: boolean;
  placeholder?: string;
  suggestions?: string[];
  showEmotionDetection?: boolean;
  showSafetyWarnings?: boolean;
  className?: string;
}

const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  disabled = false,
  placeholder = "Share your thoughts...",
  suggestions = [],
  showEmotionDetection = true,
  showSafetyWarnings = true,
  className
}) => {
  const [message, setMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [detectedEmotion, setDetectedEmotion] = useState<EmotionAnalysis | null>(null);
  const [safetyAlert, setSafetyAlert] = useState<SafetyAlert | null>(null);
  const [filteredSuggestions, setFilteredSuggestions] = useState<string[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Default suggestions
  const defaultSuggestions = [
    "I feel anxious and need some coping strategies",
    "What relaxation techniques can I try?",
    "I haven't been sleeping well lately",
    "How can I handle negative thoughts?",
    "I feel overwhelmed with work pressure",
    "Can you help me better understand my emotions?",
    "What grounding techniques are available?",
    "I'm having difficulty with mood changes"
  ];

  // Safety keyword detection
  const safetyKeywords = [
    'suicide', 'kill myself', 'hurt myself', 'end it all',
    'want to die', 'not worth living', 'can\'t go on', 'self harm'
  ];

  // Emotion detection keywords
  const emotionKeywords = {
    anxiety: ['anxious', 'worried', 'nervous', 'panic', 'fear', 'scared', 'afraid'],
    depression: ['depressed', 'sad', 'hopeless', 'empty', 'worthless', 'down'],
    anger: ['angry', 'mad', 'furious', 'irritated', 'frustrated'],
    stress: ['stressed', 'overwhelmed', 'pressure', 'burdened'],
    loneliness: ['lonely', 'alone', 'disconnected', 'isolated']
  };

  // Auto-adjust textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  }, [message]);

  // Analyze message content
  useEffect(() => {
    if (message.trim()) {
      analyzeMessage(message);
      filterSuggestions(message);
    } else {
      setDetectedEmotion(null);
      setSafetyAlert(null);
      setFilteredSuggestions([]);
    }
  }, [message]);

  // Debounced emotion detection
  const debouncedEmotionAnalysis = useCallback(
    debounce((text: string) => {
      if (!showEmotionDetection) return;
      
      const lowercaseText = text.toLowerCase();
      let detectedEmotions: string[] = [];
      let maxConfidence = 0;
      let primaryEmotion = '';

      Object.entries(emotionKeywords).forEach(([emotion, keywords]) => {
        const matches = keywords.filter(keyword => lowercaseText.includes(keyword));
        if (matches.length > 0) {
          detectedEmotions.push(emotion);
          const confidence = matches.length / keywords.length;
          if (confidence > maxConfidence) {
            maxConfidence = confidence;
            primaryEmotion = emotion;
          }
        }
      });

      if (primaryEmotion) {
        setDetectedEmotion({
          primary: primaryEmotion,
          confidence: Math.min(maxConfidence, 1),
          secondary: detectedEmotions.filter(e => e !== primaryEmotion),
          intensity: Math.min(detectedEmotions.length / 3, 1),
          suggestions: getEmotionSuggestions(primaryEmotion)
        });
      }
    }, 500),
    [showEmotionDetection]
  );

  const analyzeMessage = (text: string) => {
    const lowercaseText = text.toLowerCase();
    
    // Safety analysis
    if (showSafetyWarnings) {
      const hasSafetyRisk = safetyKeywords.some(keyword => 
        lowercaseText.includes(keyword)
      );
      
      if (hasSafetyRisk) {
        setSafetyAlert({
          level: 'critical',
          type: 'self_harm',
          message: 'Potential safety risk detected, please seek professional help',
          recommendations: [
            'Consider contacting a mental health professional',
            'Call a mental health hotline',
            'Talk to a trusted friend or family member'
          ],
          resources: [
            {
              name: 'National Suicide Prevention Lifeline',
              description: '24/7 free confidential support',
              url: 'https://988lifeline.org/',
              phone: '988',
              available: true
            }
          ]
        });
        return;
      }
    }

    // Emotion detection
    debouncedEmotionAnalysis(text);
  };

  const filterSuggestions = (text: string) => {
    if (!text.trim()) {
      setFilteredSuggestions([]);
      return;
    }

    const allSuggestions = [...defaultSuggestions, ...suggestions];
    const filtered = allSuggestions.filter(suggestion =>
      suggestion.toLowerCase().includes(text.toLowerCase())
    ).slice(0, 3);

    setFilteredSuggestions(filtered);
  };

  const getEmotionSuggestions = (emotion: string): string[] => {
    const suggestions = {
      anxiety: ['Deep breathing exercises', 'Progressive muscle relaxation', 'Mindfulness meditation'],
      depression: ['Regular exercise', 'Maintain social connections', 'Seek professional help'],
      anger: ['Count to ten', 'Deep breathing', 'Take a break from the situation'],
      stress: ['Time management', 'Set boundaries', 'Seek support'],
      loneliness: ['Join interest groups', 'Reach out to old friends', 'Volunteer']
    };
    return suggestions[emotion as keyof typeof suggestions] || [];
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    setMessage(value);
    setIsTyping(true);
    
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current);
    }
    
    typingTimeoutRef.current = setTimeout(() => {
      setIsTyping(false);
    }, 1000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    if (!message.trim() || disabled) return;

    const metadata = {
      emotion_detected: detectedEmotion,
      safety_alert: safetyAlert,
      typing_duration: Date.now() - (typingTimeoutRef.current ? 0 : Date.now())
    };

    onSend(message, metadata);
    setMessage('');
    setDetectedEmotion(null);
    setSafetyAlert(null);
    setFilteredSuggestions([]);
    
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setMessage(suggestion);
    setShowSuggestions(false);
    textareaRef.current?.focus();
  };

  const getEmotionColor = (emotion: string) => {
    const colors = {
      anxiety: 'text-warning-600 bg-warning-50',
      depression: 'text-secondary-600 bg-secondary-50',
      anger: 'text-error-600 bg-error-50',
      stress: 'text-warning-600 bg-warning-50',
      loneliness: 'text-mental-600 bg-mental-50'
    };
    return colors[emotion as keyof typeof colors] || 'text-secondary-600 bg-secondary-50';
  };

  return (
    <div className={cn('relative', className)}>
      {/* Safety alert */}
      {safetyAlert && (
        <div className="mb-4 p-4 bg-error-50 border border-error-200 rounded-xl animate-slide-down">
          <div className="flex items-start gap-3">
            <AlertTriangle className="h-5 w-5 text-error-600 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <h4 className="text-sm font-medium text-error-800 mb-2">
                {safetyAlert.message}
              </h4>
              <ul className="text-xs text-error-700 space-y-1 mb-3">
                {safetyAlert.recommendations.map((rec, index) => (
                  <li key={index}>• {rec}</li>
                ))}
              </ul>
              {safetyAlert.resources.map((resource, index) => (
                <div key={index} className="text-xs">
                  <strong>{resource.name}:</strong> {resource.description}
                  {resource.phone && (
                    <span className="ml-2 text-primary-600">{resource.phone}</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Emotion detection indicator */}
      {detectedEmotion && (
        <div className="mb-3 flex items-center gap-2 animate-slide-down">
          <div className={cn('px-2 py-1 rounded-full text-xs font-medium', getEmotionColor(detectedEmotion.primary))}>
            <Smile className="h-3 w-3 inline mr-1" />
            {detectedEmotion.primary === 'anxiety' && 'Anxiety'}
            {detectedEmotion.primary === 'depression' && 'Depression'}
            {detectedEmotion.primary === 'anger' && 'Anger'}
            {detectedEmotion.primary === 'stress' && 'Stress'}
            {detectedEmotion.primary === 'loneliness' && 'Loneliness'}
          </div>
          {detectedEmotion.suggestions.length > 0 && (
            <div className="text-xs text-secondary-500">
              Suggestion: {detectedEmotion.suggestions[0]}
            </div>
          )}
        </div>
      )}

      {/* Input area */}
      <div className="relative">
        <div className="flex items-end gap-3 p-4 bg-white border border-secondary-200 rounded-2xl shadow-soft">


          {/* Text input */}
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={handleInputChange}
              onKeyPress={handleKeyPress}
              placeholder={placeholder}
              disabled={disabled}
              className="w-full resize-none border-0 bg-transparent text-sm placeholder-secondary-500 focus:outline-none focus:ring-0 min-h-[20px] max-h-[120px]"
              rows={1}
            />
            
            {/* Typing indicator */}
            {isTyping && (
              <div className="absolute -top-6 left-0 text-xs text-secondary-400">
                Typing...
              </div>
            )}
          </div>

          {/* Voice button */}
          <button
            className={cn(
              "p-2 rounded-full transition-colors",
              isRecording 
                ? "bg-error-100 text-error-600" 
                : "text-secondary-500 hover:text-primary-500"
            )}
            onClick={() => setIsRecording(!isRecording)}
            aria-label={isRecording ? "Stop recording" : "Start recording"}
          >
            <Mic className="h-4 w-4" />
          </button>

          {/* Send button */}
          <Button
            onClick={handleSend}
            disabled={!message.trim() || disabled}
            className="rounded-full p-2"
            aria-label="Send message"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>

        {/* Suggestions list */}
        {filteredSuggestions.length > 0 && (
          <div className="absolute bottom-full left-0 right-0 mb-2 bg-white border border-secondary-200 rounded-xl shadow-medium p-2 animate-slide-down">
            <div className="text-xs text-secondary-500 mb-2 flex items-center gap-1">
              <Sparkles className="h-3 w-3" />
              Suggestions
            </div>
            <div className="space-y-1">
              {filteredSuggestions.map((suggestion, index) => (
                <button
                  key={index}
                  onClick={() => handleSuggestionClick(suggestion)}
                  className="block w-full text-left p-2 text-xs text-secondary-700 hover:bg-secondary-50 rounded-lg transition-colors"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatInput; 