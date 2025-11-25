// API Client

import { GenerateRequest, GenerateResponse } from '@/types/api';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function generateText(request: GenerateRequest): Promise<GenerateResponse> {
    const response = await fetch(`${API_BASE_URL}/generate`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
    });

    if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
    }

    return response.json();
}

export async function checkHealth() {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.json();
}

export async function getModelInfo() {
    const response = await fetch(`${API_BASE_URL}/model_info`);
    return response.json();
}
