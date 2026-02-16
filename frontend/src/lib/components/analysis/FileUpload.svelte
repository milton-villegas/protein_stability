<script lang="ts">
	import { uploadData } from '$lib/stores/analysis';
	import { uploadResults } from '$lib/api/analysis';
	import { showToast } from '$lib/stores/ui';

	let uploading = $state(false);
	let dragOver = $state(false);
	let fileInput: HTMLInputElement;

	async function handleFile(file: File) {
		if (!file.name.match(/\.xlsx?$/)) {
			showToast('Only Excel files (.xlsx, .xls) are supported', 'error');
			return;
		}
		uploading = true;
		try {
			$uploadData = await uploadResults(file);
			showToast(`Loaded ${$uploadData.total_rows} rows, ${$uploadData.columns.length} columns`, 'success');
		} catch (e: any) {
			showToast(e.message, 'error');
		}
		uploading = false;
	}

	function onDrop(e: DragEvent) {
		e.preventDefault();
		dragOver = false;
		const file = e.dataTransfer?.files[0];
		if (file) handleFile(file);
	}

	function onFileSelect(e: Event) {
		const target = e.target as HTMLInputElement;
		const file = target.files?.[0];
		if (file) handleFile(file);
	}
</script>

<div
	class="border-2 border-dashed rounded-lg p-6 text-center transition-colors cursor-pointer"
	class:border-primary={dragOver}
	class:border-base-300={!dragOver}
	class:bg-primary={dragOver}
	ondragover={(e) => { e.preventDefault(); dragOver = true; }}
	ondragleave={() => dragOver = false}
	ondrop={onDrop}
	onclick={() => fileInput?.click()}
	role="button"
	tabindex="0"
>
	{#if uploading}
		<span class="loading loading-spinner loading-md text-primary"></span>
		<p class="text-sm mt-2">Uploading...</p>
	{:else if $uploadData}
		<p class="text-sm font-medium text-success">File loaded</p>
		<p class="text-xs opacity-60">{$uploadData.total_rows} rows, {$uploadData.columns.length} columns</p>
		<p class="text-xs opacity-40 mt-1">Click or drop to replace</p>
	{:else}
		<p class="text-sm">Drop Excel file here or click to browse</p>
		<p class="text-xs opacity-60 mt-1">Accepts .xlsx and .xls files</p>
	{/if}
	<input type="file" accept=".xlsx,.xls" class="hidden" bind:this={fileInput} onchange={onFileSelect} />
</div>
