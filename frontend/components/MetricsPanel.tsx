'use client';

import { GenerateResponse } from '@/types/api';
import { useState } from 'react';
import SpeedChart from './SpeedChart';
import MemoryChart from './MemoryChart';
import QualityMetricsDisplay from './QualityMetrics';
import TopKCandidates from './TopKCandidates';
import { Zap, Database, TrendingUp, Target } from 'lucide-react';

interface MetricsPanelProps {
  metrics: GenerateResponse;
}

export default function MetricsPanel({ metrics }: MetricsPanelProps) {
  const [activeTab, setActiveTab] = useState<'speed' | 'memory' | 'quality' | 'generation'>('speed');

  const tabs = [
    { id: 'speed' as const, label: 'Speed', icon: Zap },
    { id: 'memory' as const, label: 'Memory', icon: Database },
    { id: 'quality' as const, label: 'Quality', icon: TrendingUp },
    ...(metrics.generation_steps ? [{ id: 'generation' as const, label: 'Steps', icon: Target }] : []),
  ];

  return (
    <div className="card-glass rounded-3xl p-6 shadow-2xl">
      <div className="flex gap-2 mb-6 p-1 glass rounded-2xl">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 py-2.5 px-4 rounded-xl text-sm font-medium transition-all duration-300 flex items-center justify-center gap-2 ${
                activeTab === tab.id
                  ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg'
                  : 'text-white/60 hover:text-white/80 hover:bg-white/5'
              }`}
            >
              <Icon size={16} />
              {tab.label}
            </button>
          );
        })}
      </div>

      <div className="min-h-[300px]">
        {activeTab === 'speed' && <SpeedChart metrics={metrics.speed_metrics} />}
        {activeTab === 'memory' && <MemoryChart memory={metrics.memory_usage} />}
        {activeTab === 'quality' && <QualityMetricsDisplay metrics={metrics.quality_metrics} />}
        {activeTab === 'generation' && metrics.generation_steps && (
          <TopKCandidates steps={metrics.generation_steps} />
        )}
      </div>

      <div className="mt-6 pt-6 border-t border-white/10">
        <div className="text-xs text-white/40 space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-white/60">Parameters:</span>
            <span>temp={metrics.parameters.temperature}</span>
            <span>top_k={metrics.parameters.top_k || 'null'}</span>
            <span>top_p={metrics.parameters.top_p || 'null'}</span>
            <span>rep={metrics.parameters.repetition_penalty}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-white/60">Tokens:</span>
            <span>{metrics.statistics.prompt_length} prompt</span>
            <span>+</span>
            <span>{metrics.statistics.generated_tokens} generated</span>
            <span>=</span>
            <span className="text-white/80 font-medium">{metrics.statistics.total_tokens} total</span>
          </div>
        </div>
      </div>
    </div>
  );
}
