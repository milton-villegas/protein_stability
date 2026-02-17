import { request, uploadFile, downloadFile, triggerDownload } from './client';
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

export async function getAnalysisSummary(): Promise<Record<string, any>> {
	return request('/api/analysis/summary');
}

export async function exportResults(): Promise<void> {
	const { blob, filename } = await downloadFile('/api/analysis/export/results');
	triggerDownload(blob, filename ?? 'analysis_results.xlsx');
}

export async function exportBOBatch(finalVolume: number = 100, batchNumber: number = 1): Promise<void> {
	const { blob, filename } = await downloadFile('/api/analysis/export/bo-batch', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ final_volume: finalVolume, batch_number: batchNumber }),
	});
	triggerDownload(blob, filename ?? 'bo_batch.xlsx');
}

export async function exportBOCsv(finalVolume: number = 100, batchNumber: number = 1): Promise<void> {
	const { blob, filename } = await downloadFile('/api/analysis/export/bo-csv', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ final_volume: finalVolume, batch_number: batchNumber }),
	});
	triggerDownload(blob, filename ?? 'bo_batch.csv');
}

export async function getParetoFrontier(): Promise<any> {
	return request('/api/analysis/pareto');
}
