'use client';

import { GenerationStep } from '@/types/api';
import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface TopKCandidatesProps {
  steps: GenerationStep[];
}

export default function TopKCandidates({ steps }: TopKCandidatesProps) {
  const [currentStep, setCurrentStep] = useState(0);

  if (!steps || steps.length === 0 || !steps[0].top_k_candidates) {
    return (
      <div className="glass rounded-2xl p-12 text-center">
        <div className="text-white/40 text-sm">
          No top-k candidates available. Enable &quot;Top-K Viz&quot; in settings.
        </div>
      </div>
    );
  }

  const step = steps[currentStep];
  const candidates = step.top_k_candidates || [];

  const data = candidates.map((c) => ({
    token: c.token_text.replace(/\n/g, '\\n').substring(0, 20),
    probability: c.probability * 100,
    isSelected: c.token_id === step.token_id,
  }));

  return (
    <div className="space-y-6 chart-container">
      <div className="flex items-center justify-between glass rounded-2xl p-3">
        <div className="text-sm text-white/60">
          Step <span className="text-white font-medium">{currentStep + 1}</span> of {steps.length}
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
            disabled={currentStep === 0}
            className="btn-glass p-2 rounded-xl text-white/80 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <ChevronLeft size={18} />
          </button>
          <button
            onClick={() => setCurrentStep(Math.min(steps.length - 1, currentStep + 1))}
            disabled={currentStep === steps.length - 1}
            className="btn-glass p-2 rounded-xl text-white/80 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <ChevronRight size={18} />
          </button>
        </div>
      </div>

      <div className="metric-card rounded-2xl p-5">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-2 h-2 rounded-full bg-blue-500 glow-blue" />
          <div className="text-white/60 text-xs">Selected Token</div>
        </div>
        <div className="text-2xl font-mono font-semibold text-white mb-2">{step.token_text}</div>
        <div className="text-xs text-white/40">
          Probability: {(candidates.find(c => c.token_id === step.token_id)?.probability * 100 || 0).toFixed(2)}%
        </div>
      </div>

      <div className="glass rounded-2xl p-4">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis 
              type="number" 
              stroke="rgba(255,255,255,0.5)"
              label={{ value: 'Probability (%)', position: 'insideBottom', offset: -5, fill: 'rgba(255,255,255,0.5)' }} 
            />
            <YAxis 
              type="category" 
              dataKey="token" 
              width={100} 
              stroke="rgba(255,255,255,0.5)"
            />
            <Tooltip 
              formatter={(value: number) => `${value.toFixed(2)}%`}
              contentStyle={{ 
                background: 'rgba(17, 25, 40, 0.8)', 
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '12px',
                backdropFilter: 'blur(20px)'
              }}
            />
            <Bar dataKey="probability" name="Probability (%)" radius={[0, 8, 8, 0]}>
              {data.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={entry.isSelected ? 'url(#selectedGradient)' : 'rgba(148, 163, 184, 0.6)'} 
                />
              ))}
            </Bar>
            <defs>
              <linearGradient id="selectedGradient" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#667eea" />
                <stop offset="100%" stopColor="#764ba2" />
              </linearGradient>
            </defs>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="text-xs text-white/40 text-center">
        Gradient bar indicates the selected token. Gray bars show alternative candidates.
      </div>
    </div>
  );
}
