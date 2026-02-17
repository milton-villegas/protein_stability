const BASE_URL = import.meta.env.VITE_API_URL || '';
const SESSION_HEADER = 'X-Session-ID';

let sessionId: string | null = null;

function captureSession(res: Response) {
	const id = res.headers.get(SESSION_HEADER);
	if (id) sessionId = id;
}

function sessionHeaders(): Record<string, string> {
	return sessionId ? { [SESSION_HEADER]: sessionId } : {};
}

export class ApiError extends Error {
	constructor(public status: number, message: string) {
		super(message);
	}
}

export async function request<T>(path: string, options?: RequestInit): Promise<T> {
	const res = await fetch(`${BASE_URL}${path}`, {
		headers: { 'Content-Type': 'application/json', ...sessionHeaders(), ...options?.headers },
		...options,
	});

	captureSession(res);

	if (!res.ok) {
		let detail = 'Request failed';
		try {
			const err = await res.json();
			detail = err.detail || detail;
		} catch {}
		throw new ApiError(res.status, detail);
	}

	return res.json();
}

export async function uploadFile<T>(path: string, file: File): Promise<T> {
	const form = new FormData();
	form.append('file', file);

	const res = await fetch(`${BASE_URL}${path}`, {
		method: 'POST',
		headers: { ...sessionHeaders() },
		body: form,
	});

	captureSession(res);

	if (!res.ok) {
		let detail = 'Upload failed';
		try {
			const err = await res.json();
			detail = err.detail || detail;
		} catch {}
		throw new ApiError(res.status, detail);
	}

	return res.json();
}

export async function downloadFile(path: string, options?: RequestInit): Promise<Blob> {
	const res = await fetch(`${BASE_URL}${path}`, {
		headers: { ...sessionHeaders(), ...options?.headers },
		...options,
	});

	captureSession(res);

	if (!res.ok) {
		let detail = 'Download failed';
		try {
			const err = await res.json();
			detail = err.detail || detail;
		} catch {}
		throw new ApiError(res.status, detail);
	}

	return res.blob();
}

export function triggerDownload(blob: Blob, filename: string) {
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = filename;
	a.click();
	URL.revokeObjectURL(url);
}
