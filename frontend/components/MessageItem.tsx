'use client';

import { Message } from '@/types/api';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';
import MetricsPanel from './MetricsPanel';

interface MessageItemProps {
  message: Message;
}

export default function MessageItem({ message }: MessageItemProps) {
  const [showMetrics, setShowMetrics] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const messageRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  return (
    <div 
      ref={messageRef}
      className={`mb-6 ${message.role === 'user' ? 'text-right' : 'text-left'} ${
        isVisible 
          ? message.role === 'user' ? 'message-slide-right' : 'message-slide-left' 
          : 'opacity-0'
      }`}
    >
      <div className={`inline-block max-w-[80%] ${
        message.role === 'user' 
          ? 'bg-gradient-to-br from-blue-600 to-purple-600 text-white rounded-[20px] px-5 py-3.5 shadow-lg' 
          : 'glass text-white rounded-[20px] px-5 py-3.5 shadow-lg'
      }`}>
        <div className="whitespace-pre-wrap leading-relaxed">{message.content}</div>
        {message.metrics && (
          <button
            onClick={() => setShowMetrics(!showMetrics)}
            className={`mt-3 flex items-center gap-1.5 text-xs opacity-70 hover:opacity-100 transition-all ${
              message.role === 'user' ? 'text-white' : 'text-white/80'
            }`}
          >
            {showMetrics ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            {showMetrics ? 'Hide' : 'Show'} Metrics
          </button>
        )}
      </div>
      {message.metrics && showMetrics && (
        <div className="mt-4 inline-block max-w-[90%]">
          <MetricsPanel metrics={message.metrics} />
        </div>
      )}
    </div>
  );
}
