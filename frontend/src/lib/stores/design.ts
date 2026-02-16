import { writable } from 'svelte/store';
import type { FactorsResponse, DesignGenerateResponse, AvailableFactors, DesignTypes } from '$lib/types';

export const availableFactors = writable<AvailableFactors | null>(null);
export const designTypes = writable<DesignTypes | null>(null);
export const currentFactors = writable<FactorsResponse | null>(null);
export const designResult = writable<DesignGenerateResponse | null>(null);
export const selectedDesignType = writable('full_factorial');
export const finalVolume = writable(100);
export const proteinStock = writable<number | null>(null);
export const proteinFinal = writable<number | null>(null);
