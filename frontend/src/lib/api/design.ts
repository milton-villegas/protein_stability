import { request, downloadFile, triggerDownload } from './client';
import type { FactorsResponse, DesignGenerateResponse } from '$lib/types';

export async function getFactors(): Promise<FactorsResponse> {
	return request('/api/design/factors');
}

export async function addFactor(name: string, levels: string[], stockConc?: number, perLevelConcs?: any): Promise<FactorsResponse> {
	return request('/api/design/factors', {
		method: 'POST',
		body: JSON.stringify({
			name,
			levels,
			stock_conc: stockConc ?? null,
			per_level_concs: perLevelConcs ?? null,
		}),
	});
}

export async function addFromAvailable(internalName: string, levels: string[], stockConc?: number, perLevelConcs?: any): Promise<FactorsResponse> {
	return request('/api/design/factors/from-available', {
		method: 'POST',
		body: JSON.stringify({
			internal_name: internalName,
			levels,
			stock_conc: stockConc ?? null,
			per_level_concs: perLevelConcs ?? null,
		}),
	});
}

export async function updateFactor(name: string, levels: string[], stockConc?: number, perLevelConcs?: any): Promise<FactorsResponse> {
	return request(`/api/design/factors/${encodeURIComponent(name)}`, {
		method: 'PUT',
		body: JSON.stringify({
			levels,
			stock_conc: stockConc ?? null,
			per_level_concs: perLevelConcs ?? null,
		}),
	});
}

export async function removeFactor(name: string): Promise<FactorsResponse> {
	return request(`/api/design/factors/${encodeURIComponent(name)}`, {
		method: 'DELETE',
	});
}

export async function clearFactors(): Promise<FactorsResponse> {
	return request('/api/design/factors/clear', { method: 'POST' });
}

export async function getCombinations(): Promise<{ total_combinations: number; plates_required: number }> {
	return request('/api/design/combinations');
}

export async function generateDesign(designType: string, params: Record<string, any> = {}, finalVolume: number = 200): Promise<DesignGenerateResponse> {
	return request('/api/design/generate', {
		method: 'POST',
		body: JSON.stringify({
			design_type: designType,
			final_volume: finalVolume,
			params,
		}),
	});
}

export async function buildFactorial(finalVolume: number = 200, proteinStock?: number, proteinFinal?: number): Promise<any> {
	return request('/api/design/build-factorial', {
		method: 'POST',
		body: JSON.stringify({
			final_volume: finalVolume,
			protein_stock: proteinStock ?? null,
			protein_final: proteinFinal ?? null,
		}),
	});
}

export async function exportExcel(finalVolume: number = 200, proteinStock?: number, proteinFinal?: number): Promise<void> {
	const blob = await downloadFile('/api/design/export/excel', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({
			final_volume: finalVolume,
			protein_stock: proteinStock ?? null,
			protein_final: proteinFinal ?? null,
		}),
	});
	triggerDownload(blob, 'Design.xlsx');
}

export async function exportCsv(finalVolume: number = 200, proteinStock?: number, proteinFinal?: number): Promise<void> {
	const blob = await downloadFile('/api/design/export/csv', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({
			final_volume: finalVolume,
			protein_stock: proteinStock ?? null,
			protein_final: proteinFinal ?? null,
		}),
	});
	triggerDownload(blob, 'Opentrons.csv');
}
