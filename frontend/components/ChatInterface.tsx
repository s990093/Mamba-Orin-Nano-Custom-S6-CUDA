'use client';

import { useState } from 'react';
import { Message } from '@/types/api';
import { generateText } from '@/lib/api';
import MessageItem from './MessageItem';
import GPUWidget from './GPUWidget';
import { Send, Settings, Sparkles, Sliders } from 'lucide-react';

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  
  // Generation parameters
  const [showSettings, setShowSettings] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [topK, setTopK] = useState<number | null>(50);
  const [topP, setTopP] = useState<number | null>(0.9);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.2);
  const [maxTokens, setMaxTokens] = useState(50);
  const [includeTopK, setIncludeTopK] = useState(true);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await generateText({
        prompt: input,
        temperature,
        top_k: topK,
        top_p: topP,
        repetition_penalty: repetitionPenalty,
        max_tokens: maxTokens,
        include_top_k_candidates: includeTopK,
      });

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.generated_text,
        metrics: response,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Failed to generate text'}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-blue-900/50 to-purple-900/30 -z-10" />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_120%,rgba(120,119,198,0.3),rgba(255,255,255,0))] -z-10" />
      
      {/* Header */}
      <div className="glass-dark border-b border-white/10 backdrop-blur-2xl">
        <div className="p-4 flex items-center justify-between max-w-5xl mx-auto">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center glow-blue">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-white">Mamba2 Chat</h1>
              <p className="text-xs text-white/60">Powered by MLX</p>
            </div>
          </div>
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="btn-glass p-2.5 rounded-xl text-white/80 hover:text-white flex items-center gap-2"
          >
            <Sliders size={20} />
            <span className="text-sm hidden md:inline">Parameters</span>
          </button>
        </div>
      </div>

      {/* Improved Settings Panel */}
      {showSettings && (
        <div className="glass-dark border-b border-white/10 overflow-hidden">
          <div className="p-6 max-w-5xl mx-auto">
            <div className="flex items-center gap-2 mb-6">
              <Settings className="w-5 h-5 text-white/80" />
              <h3 className="text-lg font-semibold text-white">Generation Parameters</h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Temperature */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium text-white/90">Temperature</label>
                  <span className="text-xs text-white/60 glass px-2 py-1 rounded-lg">{temperature.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  step="0.1"
                  min="0"
                  max="2"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full h-2 rounded-lg appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, #667eea ${(temperature / 2) * 100}%, rgba(255,255,255,0.1) ${(temperature / 2) * 100}%)`
                  }}
                />
                <p className="text-xs text-white/50">Controls randomness (0=deterministic, 2=very random)</p>
              </div>

              {/* Max Tokens */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium text-white/90">Max Tokens</label>
                  <span className="text-xs text-white/60 glass px-2 py-1 rounded-lg">{maxTokens}</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="500"
                  step="10"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  className="w-full h-2 rounded-lg appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, #667eea ${(maxTokens / 500) * 100}%, rgba(255,255,255,0.1) ${(maxTokens / 500) * 100}%)`
                  }}
                />
                <p className="text-xs text-white/50">Maximum length of generated response</p>
              </div>

              {/* Top-K */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium text-white/90">Top-K Sampling</label>
                  <span className="text-xs text-white/60 glass px-2 py-1 rounded-lg">{topK || 'Off'}</span>
                </div>
                <div className="flex gap-2">
                  <input
                    type="range"
                    min="0"
                    max="100"
                    step="5"
                    value={topK || 0}
                    onChange={(e) => setTopK(parseInt(e.target.value) || null)}
                    className="flex-1 h-2 rounded-lg appearance-none cursor-pointer"
                    style={{
                      background: `linear-gradient(to right, #667eea ${((topK || 0) / 100) * 100}%, rgba(255,255,255,0.1) ${((topK || 0) / 100) * 100}%)`
                    }}
                  />
                  <button
                    onClick={() => setTopK(null)}
                    className="text-xs px-2 py-1 glass rounded-lg hover:bg-white/10"
                  >
                    Off
                  </button>
                </div>
                <p className="text-xs text-white/50">Limit sampling to top K most likely tokens</p>
              </div>

              {/* Top-P */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium text-white/90">Top-P (Nucleus)</label>
                  <span className="text-xs text-white/60 glass px-2 py-1 rounded-lg">{topP?.toFixed(2) || 'Off'}</span>
                </div>
                <div className="flex gap-2">
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={topP || 0}
                    onChange={(e) => setTopP(parseFloat(e.target.value) || null)}
                    className="flex-1 h-2 rounded-lg appearance-none cursor-pointer"
                    style={{
                      background: `linear-gradient(to right, #667eea ${((topP || 0) * 100)}%, rgba(255,255,255,0.1) ${((topP || 0) * 100)}%)`
                    }}
                  />
                  <button
                    onClick={() => setTopP(null)}
                    className="text-xs px-2 py-1 glass rounded-lg hover:bg-white/10"
                  >
                    Off
                  </button>
                </div>
                <p className="text-xs text-white/50">Cumulative probability threshold for nucleus sampling</p>
              </div>

              {/* Repetition Penalty */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium text-white/90">Repetition Penalty</label>
                  <span className="text-xs text-white/60 glass px-2 py-1 rounded-lg">{repetitionPenalty.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="2"
                  step="0.1"
                  value={repetitionPenalty}
                  onChange={(e) => setRepetitionPenalty(parseFloat(e.target.value))}
                  className="w-full h-2 rounded-lg appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, #667eea ${((repetitionPenalty - 0.5) / 1.5) * 100}%, rgba(255,255,255,0.1) ${((repetitionPenalty - 0.5) / 1.5) * 100}%)`
                  }}
                />
                <p className="text-xs text-white/50">Penalize repeated tokens (&gt;1.0 reduces repetition)</p>
              </div>

              {/* Top-K Visualization Toggle */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-white/90">Visualization</label>
                <label className="flex items-center gap-3 glass p-3 rounded-xl cursor-pointer hover:bg-white/5">
                  <input
                    type="checkbox"
                    checked={includeTopK}
                    onChange={(e) => setIncludeTopK(e.target.checked)}
                    className="w-5 h-5 rounded"
                  />
                  <div>
                    <div className="text-sm text-white">Top-K Candidates Visualization</div>
                    <div className="text-xs text-white/50">Show probability distribution for each token</div>
                  </div>
                </label>
              </div>
            </div>

            {/* Quick Presets */}
            <div className="mt-6 pt-6 border-t border-white/10">
              <div className="text-sm font-medium text-white/80 mb-3">Quick Presets</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                <button
                  onClick={() => {
                    setTemperature(0.3);
                    setTopK(20);
                    setTopP(0.85);
                    setRepetitionPenalty(1.3);
                  }}
                  className="btn-glass px-4 py-2 rounded-xl text-sm text-white/90"
                >
                  🎯 Precise
                </button>
                <button
                  onClick={() => {
                    setTemperature(0.7);
                    setTopK(50);
                    setTopP(0.9);
                    setRepetitionPenalty(1.2);
                  }}
                  className="btn-glass px-4 py-2 rounded-xl text-sm text-white/90"
                >
                  ⚖️ Balanced
                </button>
                <button
                  onClick={() => {
                    setTemperature(1.2);
                    setTopK(100);
                    setTopP(0.95);
                    setRepetitionPenalty(1.1);
                  }}
                  className="btn-glass px-4 py-2 rounded-xl text-sm text-white/90"
                >
                  🎨 Creative
                </button>
                <button
                  onClick={() => {
                    setTemperature(0);
                    setTopK(null);
                    setTopP(null);
                    setRepetitionPenalty(1.0);
                  }}
                  className="btn-glass px-4 py-2 rounded-xl text-sm text-white/90"
                >
                  🔒 Greedy
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="text-center mt-32">
              <div className="inline-block mb-6">
                <div className="w-20 h-20 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center glow-blue">
                  <Sparkles className="w-10 h-10 text-white" />
                </div>
              </div>
              <h2 className="text-2xl font-semibold text-white mb-2">Start a Conversation</h2>
              <p className="text-white/60">Ask anything and explore comprehensive generation metrics</p>
            </div>
          )}
          {messages.map((message) => (
            <MessageItem key={message.id} message={message} />
          ))}
          {loading && (
            <div className="flex justify-center">
              <div className="glass px-6 py-3 rounded-full">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="w-2 h-2 rounded-full bg-purple-500 animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-2 h-2 rounded-full bg-pink-500 animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Input */}
      <div className="glass-dark border-t border-white/10 backdrop-blur-2xl">
        <div className="p-4 max-w-4xl mx-auto">
          <div className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
              placeholder="Message Mamba2..."
              className="input-glass flex-1 px-5 py-3.5 rounded-2xl text-white placeholder-white/40 focus:outline-none"
              disabled={loading}
            />
            <button
              onClick={handleSend}
              disabled={loading || !input.trim()}
              className="btn-primary px-6 py-3.5 rounded-2xl text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 shrink-0"
            >
              <Send size={18} />
            </button>
          </div>
        </div>
      </div>

      {/* GPU Monitoring Widget */}
      <GPUWidget />
    </div>
  );
}
