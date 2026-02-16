import { request, uploadFile } from './client';
import type { UploadResponse, AnalysisConfig, PlotResponse } from '$lib/types';

export async function uploadResults(file: File): Promise<UploadResponse> {
	return uploadFile('/api/analysis/upload', file);
}

export async function configureAnalysis(
	responseColumns: string[],
	directions?: Record<string, string>,
	constraints?: Record<string, Record<string, number>>,
): Promise<AnalysisConfig> {
	return request('/api/analysis/configure', {
		method: 'POST',
		body: JSON.stringify({
			response_columns: responseColumns,
			directions: directions ?? null,
			constraints: constraints ?? null,
		}),
	});
}

export async function runAnalysis(modelType: string = 'linear'): Promise<any> {
	return request('/api/analysis/run', {
		method: 'POST',
		body: JSON.stringify({ model_type: modelType }),
	});
}

export async function compareModels(): Promise<any> {
	return request('/api/analysis/compare-models', { method: 'POST' });
}

export async function getMainEffects(response?: string): Promise<any> {
	const params = response ? `?response=${encodeURIComponent(response)}` : '';
	return request(`/api/analysis/main-effects${params}`);
}

export async function getPlot(plotType: string, response?: string): Promise<PlotResponse> {
	const params = response ? `?response=${encodeURIComponent(response)}` : '';
	return request(`/api/analysis/plot/${plotType}${params}`);
}

export async function runOptimization(
	responseColumns: string[],
	directions: Record<string, string>,
	constraints?: Record<string, Record<string, number>>,
	nSuggestions: number = 5,
): Promise<any> {
	return request('/api/analysis/optimize', {
		method: 'POST',
		body: JSON.stringify({
			response_columns: responseColumns,
			directions,
			constraints: constraints ?? null,
			n_suggestions: nSuggestions,
		}),
	});
}
