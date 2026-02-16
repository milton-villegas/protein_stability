<script lang="ts">
	import { currentFactors } from '$lib/stores/design';
	import { removeFactor } from '$lib/api/design';
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
</script>

<div class="card bg-base-200 shadow">
	<div class="card-body p-4">
		<h3 class="card-title text-sm">Current Design Factors</h3>

		{#if $currentFactors && Object.keys($currentFactors.factors).length > 0}
			<div class="overflow-x-auto">
				<table class="table table-xs">
					<thead>
						<tr>
							<th>Factor</th>
							<th>Levels</th>
							<th># Levels</th>
							<th>Stock Conc</th>
							<th>Actions</th>
						</tr>
					</thead>
					<tbody>
						{#each Object.entries($currentFactors.factors) as [name, levels]}
							<tr class="hover:bg-base-300">
								<td class="font-medium text-xs">{name}</td>
								<td class="text-xs max-w-40 truncate">{levels.slice(0, 5).join(', ')}{levels.length > 5 ? '...' : ''}</td>
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
									<div class="flex gap-1">
										<button class="btn btn-xs btn-ghost" onclick={() => onEdit(name)}>Edit</button>
										<button class="btn btn-xs btn-ghost text-error" onclick={() => handleRemove(name)}>Delete</button>
									</div>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>

			<div class="flex gap-4 mt-2 text-sm">
				<span class="badge badge-primary">{$currentFactors.total_combinations} combinations</span>
				<span class="badge badge-secondary">{$currentFactors.plates_required} plate(s)</span>
			</div>
		{:else}
			<p class="text-sm opacity-60 py-4 text-center">No factors added yet. Select from the list on the left.</p>
		{/if}
	</div>
</div>
