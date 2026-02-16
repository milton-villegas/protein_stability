import { writable } from 'svelte/store';
import type { UploadResponse, ResponseConfig } from '$lib/types';

export const uploadData = writable<UploadResponse | null>(null);
export const responseConfigs = writable<ResponseConfig[]>([]);
export const selectedModelType = writable('auto');
export const useBayesian = writable(false);
export const analysisResults = writable<Record<string, any> | null>(null);
export const modelComparison = writable<any>(null);
export const plots = writable<Record<string, string>>({});
export const suggestions = writable<any[] | null>(null);
export const hasPareto = writable(false);
export const paretoPoints = writable<any[] | null>(null);
export const analysisSummary = writable<Record<string, any> | null>(null);
