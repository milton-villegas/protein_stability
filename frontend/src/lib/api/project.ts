import { request, uploadFile, downloadFile, triggerDownload } from './client';
import type { ProjectInfo } from '$lib/types';

export async function createProject(name: string = 'Untitled Project'): Promise<any> {
	return request('/api/project/new', {
		method: 'POST',
		body: JSON.stringify({ name }),
	});
}

export async function getProjectInfo(): Promise<ProjectInfo> {
	return request('/api/project/info');
}

export async function updateProjectName(name: string): Promise<any> {
	return request('/api/project/name', {
		method: 'PUT',
		body: JSON.stringify({ name }),
	});
}

export async function saveProject(): Promise<void> {
	const blob = await downloadFile('/api/project/save');
	const info = await getProjectInfo();
	triggerDownload(blob, `${info.name}.json`);
}

export async function loadProject(file: File): Promise<ProjectInfo> {
	return uploadFile('/api/project/load', file);
}
