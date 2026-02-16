<script lang="ts">
	import { finalVolume, proteinStock, proteinFinal, currentFactors, designResult, selectedDesignType } from '$lib/stores/design';
	import { buildFactorial, generateDesign, exportExcel, exportCsv } from '$lib/api/design';
	import { showToast } from '$lib/stores/ui';
	import DataTable from '$lib/components/shared/DataTable.svelte';

	interface Props {
		designParams: Record<string, any>;
	}

	let { designParams }: Props = $props();
	let generating = $state(false);
	let exporting = $state(false);

	async function handleGenerate() {
		generating = true;
		try {
			if ($selectedDesignType === 'full_factorial') {
				const result = await buildFactorial($finalVolume, $proteinStock ?? undefined, $proteinFinal ?? undefined);
				$designResult = {
					design_points: result.excel_data,
					total_runs: result.total_runs,
					plates_required: result.plates_required,
					warnings: result.warnings,
				};
			} else {
				$designResult = await generateDesign($selectedDesignType, designParams, $finalVolume);
			}
			showToast(`Design generated: ${$designResult?.total_runs} runs`, 'success');
		} catch (e: any) {
			showToast(e.message, 'error');
		}
		generating = false;
	}

	async function handleExportExcel() {
		exporting = true;
		try {
			await exportExcel($finalVolume);
			showToast('Excel exported', 'success');
		} catch (e: any) {
			showToast(e.message, 'error');
		}
		exporting = false;
	}

	async function handleExportCsv() {
		exporting = true;
		try {
			await exportCsv($finalVolume);
			showToast('CSV exported', 'success');
		} catch (e: any) {
			showToast(e.message, 'error');
		}
		exporting = false;
	}

	let hasFactors = $derived($currentFactors && Object.keys($currentFactors.factors).length > 0);
</script>

<div class="card bg-base-200 shadow">
	<div class="card-body p-4">
		<h3 class="card-title text-sm">Settings & Export</h3>

		<div class="grid grid-cols-2 gap-3">
			<div class="form-control">
				<label class="label"><span class="label-text text-xs">Final Volume (uL)</span></label>
				<input type="number" class="input input-sm input-bordered" bind:value={$finalVolume} min="1" max="323" />
			</div>
			<div class="form-control">
				<label class="label"><span class="label-text text-xs">Protein Stock (mg/mL)</span></label>
				<input type="number" class="input input-sm input-bordered" bind:value={$proteinStock} step="any" placeholder="Optional" />
			</div>
			<div class="form-control">
				<label class="label"><span class="label-text text-xs">Final Protein (mg/mL)</span></label>
				<input type="number" class="input input-sm input-bordered" bind:value={$proteinFinal} step="any" placeholder="Optional" />
			</div>
		</div>

		<div class="flex gap-2 mt-3">
			<button class="btn btn-sm btn-primary flex-1" onclick={handleGenerate} disabled={!hasFactors || generating}>
				{#if generating}
					<span class="loading loading-spinner loading-xs"></span>
				{/if}
				Generate Design
			</button>
		</div>

		{#if $designResult}
			<div class="flex gap-2 mt-2">
				<button class="btn btn-sm btn-outline flex-1" onclick={handleExportExcel} disabled={exporting}>
					Export Excel
				</button>
				<button class="btn btn-sm btn-outline flex-1" onclick={handleExportCsv} disabled={exporting}>
					Export CSV (Opentrons)
				</button>
			</div>

			<div class="mt-3">
				<div class="flex gap-2 mb-2">
					<span class="badge badge-primary">{$designResult.total_runs} runs</span>
					<span class="badge badge-secondary">{$designResult.plates_required} plate(s)</span>
					{#each $designResult.warnings as w}
						<span class="badge badge-warning text-xs">{w}</span>
					{/each}
				</div>
				<DataTable data={$designResult.design_points} maxRows={50} />
			</div>
		{/if}
	</div>
</div>
