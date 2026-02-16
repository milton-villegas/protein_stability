import { writable } from 'svelte/store';
import type { ProjectInfo } from '$lib/types';

export const projectInfo = writable<ProjectInfo | null>(null);
export const sessionActive = writable(false);
