'use client';

import { MemoryUsage } from '@/types/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface MemoryChartProps {
  memory: MemoryUsage;
}

export default function MemoryChart({ memory }: MemoryChartProps) {
  if (!memory.tracking_available) {
    return (
      <div className="glass rounded-2xl p-12 text-center">
        <div className="text-white/40 text-sm">Memory tracking not available</div>
      </div>
    );
  }

  const data = [
    { name: 'Initial', value: memory.initial_memory },
    { name: 'Current', value: memory.current_memory },
    { name: 'Peak', value: memory.peak_memory },
  ];

  return (
    <div className="space-y-6 chart-container">
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div className="metric-card rounded-2xl p-4 col-span-2">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2 h-2 rounded-full bg-white/60" />
            <div className="text-white/60 text-xs">Device</div>
          </div>
          <div className="text-2xl font-semibold text-white">{memory.device_type}</div>
        </div>
        <div className="metric-card rounded-2xl p-4">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2 h-2 rounded-full bg-red-500 glow-pink" />
            <div className="text-white/60 text-xs">Memory Used</div>
          </div>
          <div className="text-2xl font-semibold text-white">{memory.memory_used.toFixed(2)}</div>
          <div className="text-xs text-white/40 mt-1">MB</div>
        </div>
        <div className="metric-card rounded-2xl p-4">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2 h-2 rounded-full bg-orange-500 glow-pink" />
            <div className="text-white/60 text-xs">Peak Memory</div>
          </div>
          <div className="text-2xl font-semibold text-white">{memory.peak_memory.toFixed(2)}</div>
          <div className="text-xs text-white/40 mt-1">MB</div>
        </div>
      </div>

      <div className="glass rounded-2xl p-4">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis dataKey="name" stroke="rgba(255,255,255,0.5)" />
            <YAxis 
              stroke="rgba(255,255,255,0.5)" 
              label={{ value: 'MB', angle: -90, position: 'insideLeft', fill: 'rgba(255,255,255,0.5)' }} 
            />
            <Tooltip 
              contentStyle={{ 
                background: 'rgba(17, 25, 40, 0.8)', 
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '12px',
                backdropFilter: 'blur(20px)'
              }}
            />
            <Legend />
            <Bar dataKey="value" fill="url(#memoryGradient)" name="Memory (MB)" radius={[8, 8, 0, 0]} />
            <defs>
              <linearGradient id="memoryGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ef4444" />
                <stop offset="100%" stopColor="#dc2626" />
              </linearGradient>
            </defs>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
