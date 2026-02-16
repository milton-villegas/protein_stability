<script lang="ts">
	import FileUpload from '$lib/components/analysis/FileUpload.svelte';
	import ResponseSelector from '$lib/components/analysis/ResponseSelector.svelte';
	import RunAnalysis from '$lib/components/analysis/RunAnalysis.svelte';
	import DataTable from '$lib/components/shared/DataTable.svelte';
	import { uploadData } from '$lib/stores/analysis';
</script>

<div class="flex flex-col gap-4">
	<!-- Upload Section -->
	<div class="card bg-base-200 shadow">
		<div class="card-body p-4">
			<h3 class="card-title text-sm">Load Results</h3>
			<FileUpload />
		</div>
	</div>

	{#if $uploadData}
		<!-- Data Preview -->
		<div class="card bg-base-200 shadow">
			<div class="card-body p-4">
				<h3 class="card-title text-sm">Data Preview</h3>
				<DataTable data={$uploadData.preview_rows} maxRows={10} />
			</div>
		</div>

		<!-- Response Selection -->
		<ResponseSelector />

		<!-- Analysis Controls + Results -->
		<RunAnalysis />
	{/if}
</div>
