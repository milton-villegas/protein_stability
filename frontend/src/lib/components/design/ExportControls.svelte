<script lang="ts">
	import { finalVolume, proteinStock, proteinFinal, currentFactors, designResult, selectedDesignType } from '$lib/stores/design';
	import { buildFactorial, generateDesign, exportExcel, exportCsv } from '$lib/api/design';
	import { showToast } from '$lib/stores/ui';

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

		<div class="flex flex-wrap gap-3 items-end">
			<div class="form-control">
				<label class="label py-0"><span class="label-text text-xs">Final Volume (uL)</span></label>
				<input type="number" class="input input-sm input-bordered w-28" bind:value={$finalVolume} min="1" max="323" />
			</div>
			<div class="form-control">
				<label class="label py-0"><span class="label-text text-xs">Protein Stock (mg/mL)</span></label>
				<input type="number" class="input input-sm input-bordered w-32" bind:value={$proteinStock} step="any" placeholder="Optional" />
			</div>
			<div class="form-control">
				<label class="label py-0"><span class="label-text text-xs">Final Protein (mg/mL)</span></label>
				<input type="number" class="input input-sm input-bordered w-32" bind:value={$proteinFinal} step="any" placeholder="Optional" />
			</div>

			<button class="btn btn-sm btn-primary" onclick={handleGenerate} disabled={!hasFactors || generating}>
				{#if generating}
					<span class="loading loading-spinner loading-xs"></span>
				{/if}
				Generate
			</button>

			{#if $designResult}
				<button class="btn btn-sm btn-outline" onclick={handleExportExcel} disabled={exporting}>
					Export Excel
				</button>
				<button class="btn btn-sm btn-outline" onclick={handleExportCsv} disabled={exporting}>
					Export CSV
				</button>
			{/if}
		</div>
	</div>
</div>
