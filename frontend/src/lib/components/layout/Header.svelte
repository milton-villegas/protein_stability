<script lang="ts">
	import { projectInfo } from '$lib/stores/project';
	import { createProject, saveProject, loadProject } from '$lib/api/project';
	import { sessionActive } from '$lib/stores/project';
	import { showToast } from '$lib/stores/ui';

	let fileInput: HTMLInputElement;

	async function handleNew() {
		const name = prompt('Project name:', 'Untitled Project');
		if (name !== null) {
			await createProject(name);
			sessionActive.set(true);
			projectInfo.set({ name, has_design: false, has_results: false, factors_count: 0, design_runs: null });
			showToast('New project created', 'success');
			window.location.reload();
		}
	}

	async function handleSave() {
		try {
			await saveProject();
			showToast('Project saved', 'success');
		} catch (e: any) {
			showToast(e.message, 'error');
		}
	}

	async function handleLoad() {
		fileInput?.click();
	}

	async function onFileSelected(e: Event) {
		const target = e.target as HTMLInputElement;
		const file = target.files?.[0];
		if (!file) return;
		try {
			const info = await loadProject(file);
			projectInfo.set(info);
			showToast('Project loaded', 'success');
			window.location.reload();
		} catch (err: any) {
			showToast(err.message, 'error');
		}
	}
</script>

<div class="navbar bg-base-200 shadow-sm px-4">
	<div class="flex-1 gap-3">
		<span class="text-xl font-bold tracking-tight">SCOUT</span>
		<span class="text-sm opacity-60">Screening & Condition Optimization Utility Tool</span>
	</div>
	<div class="flex-none gap-2">
		<button class="btn btn-sm btn-ghost" onclick={handleNew}>New</button>
		<button class="btn btn-sm btn-ghost" onclick={handleSave}>Save</button>
		<button class="btn btn-sm btn-ghost" onclick={handleLoad}>Open</button>
		<input type="file" accept=".json" class="hidden" bind:this={fileInput} onchange={onFileSelected} />
		{#if $projectInfo}
			<div class="badge badge-outline">{$projectInfo.name}</div>
		{/if}
	</div>
</div>
