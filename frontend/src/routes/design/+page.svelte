<script lang="ts">
	import FactorList from '$lib/components/design/FactorList.svelte';
	import SelectedFactors from '$lib/components/design/SelectedFactors.svelte';
	import FactorEditDialog from '$lib/components/design/FactorEditDialog.svelte';
	import DesignTypeSelector from '$lib/components/design/DesignTypeSelector.svelte';
	import ExportControls from '$lib/components/design/ExportControls.svelte';
	import { currentFactors } from '$lib/stores/design';
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

<div class="grid grid-cols-12 gap-4">
	<!-- Left: Available Factors -->
	<div class="col-span-3">
		<FactorList onSelect={handleSelectAvailable} />
	</div>

	<!-- Right: Design Configuration -->
	<div class="col-span-9 flex flex-col gap-4">
		<SelectedFactors onEdit={handleEditFactor} />

		<div class="grid grid-cols-2 gap-4">
			<DesignTypeSelector bind:designParams />
			<ExportControls {designParams} />
		</div>
	</div>
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
