// API Type Definitions

export interface TopKCandidate {
    token_id: number;
    token_text: string;
    probability: number;
    log_probability: number;
}

export interface GenerationStep {
    step: number;
    token_id: number;
    token_text: string;
    log_probability: number;
    top_k_candidates?: TopKCandidate[];
}

export interface GenerationStatistics {
    prompt_length: number;
    generated_tokens: number;
    total_tokens: number;
}

export interface SpeedMetrics {
    prefill_time: number;
    prefill_speed: number;
    decode_time: number;
    decode_speed: number;
    avg_latency: number;
    total_time: number;
}

export interface MemoryUsage {
    device_type: string;
    initial_memory: number;
    current_memory: number;
    peak_memory: number;
    memory_used: number;
    tracking_available: boolean;
}

export interface QualityMetrics {
    avg_log_prob: number;
    perplexity: number;
    num_repeats: number;
    most_repeated: Array<{
        token: string;
        count: number;
    }>;
}

export interface GenerateResponse {
    generated_text: string;
    prompt: string;
    generated_only: string;
    generation_steps?: GenerationStep[];
    statistics: GenerationStatistics;
    speed_metrics: SpeedMetrics;
    memory_usage: MemoryUsage;
    quality_metrics: QualityMetrics;
    parameters: {
        temperature: number;
        top_k: number | null;
        top_p: number | null;
        repetition_penalty: number;
        max_tokens: number;
    };
}

export interface GenerateRequest {
    prompt: string;
    max_tokens?: number;
    temperature?: number;
    top_k?: number | null;
    top_p?: number | null;
    repetition_penalty?: number;
    include_top_k_candidates?: boolean;
}

export interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    metrics?: GenerateResponse;
    timestamp: Date;
}
