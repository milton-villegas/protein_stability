import { writable } from 'svelte/store';

export const loading = writable(false);
export const error = writable<string | null>(null);
export const toast = writable<{ message: string; type: 'success' | 'error' | 'info' } | null>(null);

export function showToast(message: string, type: 'success' | 'error' | 'info' = 'info') {
	toast.set({ message, type });
	setTimeout(() => toast.set(null), 4000);
}
