import { request } from './client';
import type { AvailableFactors, DesignTypes } from '$lib/types';

export async function getAvailableFactors(): Promise<AvailableFactors> {
	return request('/api/config/factors');
}

export async function getDesignTypes(): Promise<DesignTypes> {
	return request('/api/config/design-types');
}

export async function getConstraints(): Promise<any> {
	return request('/api/config/constraints');
}

export async function getConstants(): Promise<any> {
	return request('/api/config/constants');
}
