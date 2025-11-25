'use client';

import { useEffect, useState, useRef } from 'react';
import { Activity, ChevronDown, ChevronUp, Maximize2, Minimize2, GripVertical } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface GPUStatus {
  memory_allocated: number;
  memory_total: number;
  utilization: number;
}

interface HistoryPoint {
  time: string;
  memory: number;
  utilization: number;
}

export default function GPUWidget() {
  const [status, setStatus] = useState<GPUStatus>({
    memory_allocated: 0,
    memory_total: 0,
    utilization: 0,
  });
  const [collapsed, setCollapsed] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [connected, setConnected] = useState(false);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  
  // Dragging state
  const [isDragging, setIsDragging] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const widgetRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Create WebSocket connection
    const wsUrl = process.env.NEXT_PUBLIC_API_URL?.replace('http', 'ws') || 'ws://localhost:8000';
    const socket = new WebSocket(`${wsUrl}/ws/gpu`);

    socket.onopen = () => {
      console.log('GPU monitor WebSocket connected');
      setConnected(true);
    };

    socket.onmessage = (event) => {
      const data: GPUStatus = JSON.parse(event.data);
      setStatus(data);
      
      // Add to history (keep last 60 points = 60 seconds)
      const now = new Date();
      const timeStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
      
      setHistory(prev => {
        const newHistory = [
          ...prev,
          {
            time: timeStr,
            memory: data.memory_total > 0 ? (data.memory_allocated / data.memory_total) * 100 : 0,
            utilization: data.utilization
          }
        ];
        // Keep only last 60 points
        return newHistory.slice(-60);
      });
    };

    socket.onerror = (error) => {
      console.error('GPU monitor WebSocket error:', error);
      setConnected(false);
    };

    socket.onclose = () => {
      console.log('GPU monitor WebSocket disconnected');
      setConnected(false);
    };

    // Cleanup on unmount
    return () => {
      socket.close();
    };
  }, []);

  // Drag handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('.drag-handle')) {
      setIsDragging(true);
      setDragStart({
        x: e.clientX - position.x,
        y: e.clientY - position.y
      });
    }
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (isDragging) {
      setPosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, dragStart]);

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const gb = bytes / (1024 ** 3);
    const mb = bytes / (1024 ** 2);
    return gb >= 1 ? `${gb.toFixed(2)} GB` : `${mb.toFixed(0)} MB`;
  };

  const getUtilizationColor = (util: number) => {
    if (util < 30) return 'from-green-500 to-emerald-500';
    if (util < 70) return 'from-yellow-500 to-orange-500';
    return 'from-red-500 to-rose-500';
  };

  const widgetStyle = {
    transform: `translate(${position.x}px, ${position.y}px)`,
    width: expanded ? '600px' : '280px',
    cursor: isDragging ? 'grabbing' : 'default'
  };

  return (
    <div 
      ref={widgetRef}
      className="fixed bottom-6 right-6 z-50 transition-all duration-300"
      style={widgetStyle}
      onMouseDown={handleMouseDown}
    >
      <div className="glass-dark rounded-2xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="px-4 py-3 flex items-center justify-between border-b border-white/10">
          <div className="flex items-center gap-2">
            <div className="drag-handle cursor-grab active:cursor-grabbing">
              <GripVertical size={16} className="text-white/40" />
            </div>
            <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500 glow-blue' : 'bg-gray-500'}`} />
            <Activity size={16} className="text-white/80" />
            <span className="text-sm font-medium text-white">GPU Monitor</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setExpanded(!expanded)}
              className="p-1.5 rounded-lg hover:bg-white/10 transition-colors"
            >
              {expanded ? <Minimize2 size={14} className="text-white/60" /> : <Maximize2 size={14} className="text-white/60" />}
            </button>
            <button
              onClick={() => setCollapsed(!collapsed)}
              className="p-1.5 rounded-lg hover:bg-white/10 transition-colors"
            >
              {collapsed ? <ChevronUp size={14} className="text-white/60" /> : <ChevronDown size={14} className="text-white/60" />}
            </button>
          </div>
        </div>

        {/* Content */}
        {!collapsed && (
          <div className="p-4 space-y-4">
            {/* Current Stats */}
            <div className="space-y-3">
              {/* Memory Usage */}
              <div>
                <div className="flex items-center justify-between text-xs text-white/60 mb-1">
                  <span>Memory</span>
                  <span>{formatBytes(status.memory_allocated)} / {formatBytes(status.memory_total)}</span>
                </div>
                <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                  <div
                    className={`h-full bg-gradient-to-r ${getUtilizationColor(status.utilization)} transition-all duration-500`}
                    style={{ width: `${status.memory_total > 0 ? (status.memory_allocated / status.memory_total) * 100 : 0}%` }}
                  />
                </div>
              </div>

              {/* Utilization */}
              <div>
                <div className="flex items-center justify-between text-xs text-white/60 mb-1">
                  <span>Utilization</span>
                  <span>{status.utilization}%</span>
                </div>
                <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                  <div
                    className={`h-full bg-gradient-to-r ${getUtilizationColor(status.utilization)} transition-all duration-500`}
                    style={{ width: `${status.utilization}%` }}
                  />
                </div>
              </div>
            </div>

            {/* History Chart (only when expanded) */}
            {expanded && history.length > 0 && (
              <div className="glass rounded-xl p-3">
                <div className="text-xs text-white/60 mb-2">Last 60 seconds</div>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={history}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis 
                      dataKey="time" 
                      stroke="rgba(255,255,255,0.5)" 
                      tick={{ fontSize: 10 }}
                      interval={14}
                    />
                    <YAxis 
                      stroke="rgba(255,255,255,0.5)" 
                      tick={{ fontSize: 10 }}
                      domain={[0, 100]}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        background: 'rgba(17, 25, 40, 0.9)', 
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '8px',
                        fontSize: '12px'
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                    <Line 
                      type="monotone" 
                      dataKey="utilization" 
                      stroke="#3b82f6" 
                      strokeWidth={2}
                      dot={false}
                      name="GPU %"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="memory" 
                      stroke="#ef4444" 
                      strokeWidth={2}
                      dot={false}
                      name="Memory %"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Status */}
            <div className="pt-2 border-t border-white/10 text-xs text-white/40">
              {connected ? (
                <span className="flex items-center gap-1">
                  <div className="w-1 h-1 rounded-full bg-green-500" />
                  Connected • {history.length}s of data
                </span>
              ) : (
                <span className="flex items-center gap-1">
                  <div className="w-1 h-1 rounded-full bg-gray-500" />
                  Disconnected
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
