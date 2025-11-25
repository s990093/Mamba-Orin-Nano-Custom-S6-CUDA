'use client';

import { QualityMetrics } from '@/types/api';

interface QualityMetricsProps {
  metrics: QualityMetrics;
}

export default function QualityMetricsDisplay({ metrics }: QualityMetricsProps) {
  return (
    <div className="space-y-6 chart-container">
      <div className="grid grid-cols-2 gap-4">
        <div className="metric-card rounded-2xl p-5">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-2 h-2 rounded-full bg-purple-500 glow-purple" />
            <div className="text-white/60 text-xs">Perplexity</div>
          </div>
          <div className="text-3xl font-semibold text-white mb-2">{metrics.perplexity.toFixed(2)}</div>
          <div className="text-xs text-white/40">Lower is better</div>
        </div>
        <div className="metric-card rounded-2xl p-5">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-2 h-2 rounded-full bg-blue-500 glow-blue" />
            <div className="text-white/60 text-xs">Avg Log Prob</div>
          </div>
          <div className="text-3xl font-semibold text-white mb-2">{metrics.avg_log_prob.toFixed(4)}</div>
          <div className="text-xs text-white/40">Confidence metric</div>
        </div>
      </div>

      <div className="metric-card rounded-2xl p-5">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-2 h-2 rounded-full bg-yellow-500" />
          <div className="text-white/60 text-xs">Repetitions</div>
        </div>
        <div className="text-2xl font-semibold text-white mb-4">{metrics.num_repeats} tokens</div>
        {metrics.most_repeated.length > 0 && (
          <div className="space-y-3">
            <div className="text-xs text-white/50">Most Repeated:</div>
            <div className="space-y-2">
              {metrics.most_repeated.map((item, idx) => (
                <div key={idx} className="glass rounded-xl px-3 py-2 flex items-center justify-between">
                  <span className="font-mono text-sm text-white">{item.token}</span>
                  <span className="text-xs text-white/60 glass px-2 py-1 rounded-lg">
                    ×{item.count}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
