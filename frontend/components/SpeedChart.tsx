'use client';

import { SpeedMetrics } from '@/types/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface SpeedChartProps {
  metrics: SpeedMetrics;
}

export default function SpeedChart({ metrics }: SpeedChartProps) {
  const data = [
    {
      name: 'Prefill',
      speed: metrics.prefill_speed,
      time: metrics.prefill_time,
    },
    {
      name: 'Decode',
      speed: metrics.decode_speed,
      time: metrics.decode_time,
    },
  ];

  return (
    <div className="space-y-6 chart-container">
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div className="metric-card rounded-2xl p-4">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2 h-2 rounded-full bg-blue-500 glow-blue" />
            <div className="text-white/60 text-xs">Prefill</div>
          </div>
          <div className="text-2xl font-semibold text-white">{metrics.prefill_speed.toFixed(2)}</div>
          <div className="text-xs text-white/40 mt-1">tok/s • {metrics.prefill_time.toFixed(3)}s</div>
        </div>
        <div className="metric-card rounded-2xl p-4">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2 h-2 rounded-full bg-green-500 glow-blue" />
            <div className="text-white/60 text-xs">Decode</div>
          </div>
          <div className="text-2xl font-semibold text-white">{metrics.decode_speed.toFixed(2)}</div>
          <div className="text-xs text-white/40 mt-1">tok/s • {metrics.decode_time.toFixed(3)}s</div>
        </div>
        <div className="metric-card rounded-2xl p-4">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2 h-2 rounded-full bg-purple-500 glow-purple" />
            <div className="text-white/60 text-xs">Avg Latency</div>
          </div>
          <div className="text-2xl font-semibold text-white">{metrics.avg_latency.toFixed(2)}</div>
          <div className="text-xs text-white/40 mt-1">ms/token</div>
        </div>
        <div className="metric-card rounded-2xl p-4">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2 h-2 rounded-full bg-pink-500 glow-pink" />
            <div className="text-white/60 text-xs">Total Time</div>
          </div>
          <div className="text-2xl font-semibold text-white">{metrics.total_time.toFixed(3)}</div>
          <div className="text-xs text-white/40 mt-1">seconds</div>
        </div>
      </div>

      <div className="glass rounded-2xl p-4">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis dataKey="name" stroke="rgba(255,255,255,0.5)" />
            <YAxis stroke="rgba(255,255,255,0.5)" />
            <Tooltip 
              contentStyle={{ 
                background: 'rgba(17, 25, 40, 0.8)', 
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '12px',
                backdropFilter: 'blur(20px)'
              }}
            />
            <Legend />
            <Bar dataKey="speed" fill="url(#colorGradient)" name="Speed (tok/s)" radius={[8, 8, 0, 0]} />
            <defs>
              <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#667eea" />
                <stop offset="100%" stopColor="#764ba2" />
              </linearGradient>
            </defs>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
