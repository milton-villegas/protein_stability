<script lang="ts">
	import { currentFactors } from '$lib/stores/design';
	import { removeFactor, clearFactors } from '$lib/api/design';
	import { showToast } from '$lib/stores/ui';

	interface Props {
		onEdit: (name: string) => void;
	}

	let { onEdit }: Props = $props();

	async function handleRemove(name: string) {
		try {
			$currentFactors = await removeFactor(name);
			showToast(`Removed ${name}`, 'info');
		} catch (e: any) {
			showToast(e.message, 'error');
		}
	}

	async function handleClearAll() {
		try {
			$currentFactors = await clearFactors();
			showToast('All factors cleared', 'info');
		} catch (e: any) {
			showToast(e.message, 'error');
		}
	}
</script>

<div class="flex flex-col h-full">
	<div class="flex items-center justify-between mb-1">
		<h3 class="font-semibold text-sm">Current Design Factors</h3>
		<div class="flex items-center gap-2">
			{#if $currentFactors && Object.keys($currentFactors.factors).length > 0}
				<span class="badge badge-sm badge-primary">{$currentFactors.total_combinations} combinations</span>
				<span class="badge badge-sm badge-secondary">{$currentFactors.plates_required} plate(s)</span>
				<button class="btn btn-xs btn-ghost text-error" onclick={handleClearAll}>Clear All</button>
			{/if}
		</div>
	</div>

	{#if $currentFactors && Object.keys($currentFactors.factors).length > 0}
		<div class="flex-1 overflow-y-auto min-h-0">
			<table class="table table-xs table-pin-rows">
				<thead>
					<tr>
						<th class="bg-base-300 text-xs">Factor</th>
						<th class="bg-base-300 text-xs">Levels</th>
						<th class="bg-base-300 text-xs">#</th>
						<th class="bg-base-300 text-xs">Stock</th>
						<th class="bg-base-300 text-xs"></th>
					</tr>
				</thead>
				<tbody>
					{#each Object.entries($currentFactors.factors) as [name, levels]}
						<tr class="hover:bg-base-300">
							<td class="font-medium text-xs">{name}</td>
							<td class="text-xs max-w-48 truncate">{levels.slice(0, 5).join(', ')}{levels.length > 5 ? '...' : ''}</td>
							<td class="text-xs">{levels.length}</td>
							<td class="text-xs">
								{#if $currentFactors.per_level_concs[name] && Object.keys($currentFactors.per_level_concs[name]).length > 0}
									<span class="badge badge-xs badge-info">Per-level</span>
								{:else if $currentFactors.stock_concs[name]}
									{$currentFactors.stock_concs[name]}
								{:else}
									<span class="opacity-40">—</span>
								{/if}
							</td>
							<td>
								<div class="flex gap-0.5">
									<button class="btn btn-xs btn-ghost px-1" onclick={() => onEdit(name)}>Edit</button>
									<button class="btn btn-xs btn-ghost px-1 text-error" onclick={() => handleRemove(name)}>Del</button>
								</div>
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{:else}
		<div class="flex-1 flex items-center justify-center">
			<p class="text-sm opacity-60">No factors added yet. Select from the list on the left.</p>
		</div>
	{/if}
</div>
