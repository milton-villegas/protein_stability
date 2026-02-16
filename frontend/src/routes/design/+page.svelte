<script lang="ts">
	import FactorList from '$lib/components/design/FactorList.svelte';
	import SelectedFactors from '$lib/components/design/SelectedFactors.svelte';
	import FactorEditDialog from '$lib/components/design/FactorEditDialog.svelte';
	import DesignTypeSelector from '$lib/components/design/DesignTypeSelector.svelte';
	import ExportControls from '$lib/components/design/ExportControls.svelte';
	import DataTable from '$lib/components/shared/DataTable.svelte';
	import { currentFactors, designResult } from '$lib/stores/design';
	import { addFromAvailable, addFactor, updateFactor, getFactors } from '$lib/api/design';
	import { showToast } from '$lib/stores/ui';
	import { onMount } from 'svelte';

	let dialogOpen = $state(false);
	let editingFactor = $state('');
	let isNewFactor = $state(true);
	let editLevels = $state<string[]>([]);
	let editStockConc = $state<number | null>(null);
	let designParams = $state<Record<string, any>>({});

	onMount(async () => {
		try {
			$currentFactors = await getFactors();
		} catch {}
	});

	function handleSelectAvailable(key: string, display: string) {
		editingFactor = key;
		isNewFactor = true;
		editLevels = [];
		editStockConc = null;
		dialogOpen = true;
	}

	function handleEditFactor(name: string) {
		editingFactor = name;
		isNewFactor = false;
		editLevels = $currentFactors?.factors[name] ?? [];
		editStockConc = $currentFactors?.stock_concs[name] ?? null;
		dialogOpen = true;
	}

	async function handleSaveFactor(name: string, levels: string[], stockConc: number | null, perLevelConcs: any) {
		try {
			if (isNewFactor) {
				if (editingFactor === 'custom') {
					$currentFactors = await addFactor(name, levels, stockConc ?? undefined, perLevelConcs);
				} else {
					$currentFactors = await addFromAvailable(name, levels, stockConc ?? undefined, perLevelConcs);
				}
				showToast(`Added ${name}`, 'success');
			} else {
				$currentFactors = await updateFactor(name, levels, stockConc ?? undefined, perLevelConcs);
				showToast(`Updated ${name}`, 'success');
			}
			dialogOpen = false;
		} catch (e: any) {
			showToast(e.message, 'error');
		}
	}
</script>

<div class="flex flex-col gap-3" style="height: calc(100vh - 10rem);">
	<!-- Top: Factors panel (fixed height, scrollable inside) -->
	<div class="grid grid-cols-12 gap-3 min-h-0" style="height: 45%;">
		<div class="col-span-3 card bg-base-200 shadow p-3 overflow-hidden">
			<FactorList onSelect={handleSelectAvailable} />
		</div>
		<div class="col-span-9 card bg-base-200 shadow p-3 overflow-hidden">
			<SelectedFactors onEdit={handleEditFactor} />
		</div>
	</div>

	<!-- Middle: Design Type + Settings -->
	<div class="grid grid-cols-12 gap-3 shrink-0">
		<div class="col-span-3">
			<DesignTypeSelector bind:designParams />
		</div>
		<div class="col-span-9">
			<ExportControls {designParams} />
		</div>
	</div>

	<!-- Bottom: Generated design table (fills remaining space) -->
	{#if $designResult}
		<div class="card bg-base-200 shadow p-3 flex-1 min-h-0 flex flex-col overflow-hidden">
			<div class="flex items-center gap-3 mb-2 shrink-0">
				<h3 class="font-semibold text-sm">Generated Design</h3>
				<span class="badge badge-sm badge-primary">{$designResult.total_runs} runs</span>
				<span class="badge badge-sm badge-secondary">{$designResult.plates_required} plate(s)</span>
				{#each $designResult.warnings as w}
					<span class="badge badge-sm badge-warning">{w}</span>
				{/each}
			</div>
			<div class="flex-1 min-h-0">
				<DataTable data={$designResult.design_points} maxRows={100} />
			</div>
		</div>
	{/if}
</div>

<FactorEditDialog
	open={dialogOpen}
	factorName={editingFactor}
	isNew={isNewFactor}
	initialLevels={editLevels}
	initialStockConc={editStockConc}
	onSave={handleSaveFactor}
	onClose={() => dialogOpen = false}
/>
