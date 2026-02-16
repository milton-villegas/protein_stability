<script lang="ts">
	import { availableFactors } from '$lib/stores/design';
	import { getAvailableFactors } from '$lib/api/config';
	import { onMount } from 'svelte';

	interface Props {
		onSelect: (name: string, displayName: string) => void;
	}

	let { onSelect }: Props = $props();

	onMount(async () => {
		if (!$availableFactors) {
			$availableFactors = await getAvailableFactors();
		}
	});
</script>

<div class="flex flex-col h-full">
	<div class="flex items-center justify-between mb-1">
		<h3 class="font-semibold text-sm">Available Factors</h3>
	</div>
	<p class="text-xs opacity-60 mb-2">Click to add</p>
	<div class="flex-1 overflow-y-auto flex flex-col gap-0 min-h-0">
		{#if $availableFactors}
			{#if $availableFactors.factor_categories}
				{#each $availableFactors.factor_categories as category}
					<div class="text-[10px] font-bold opacity-50 uppercase tracking-wider px-2 pt-2 pb-0.5">
						{category.name}
					</div>
					{#each category.factors as key}
						{@const display = $availableFactors.factors[key] ?? key}
						<button
							class="btn btn-xs btn-ghost justify-start text-left font-normal h-7 min-h-0"
							onclick={() => onSelect(key, display)}
						>
							<span class="text-xs">{display}</span>
							{#if $availableFactors.categorical_factors.includes(key)}
								<span class="badge badge-xs badge-outline ml-auto">cat</span>
							{/if}
						</button>
					{/each}
				{/each}
			{:else}
				{#each Object.entries($availableFactors.factors) as [key, display]}
					<button
						class="btn btn-xs btn-ghost justify-start text-left font-normal h-7 min-h-0"
						onclick={() => onSelect(key, display)}
					>
						<span class="text-xs">{display}</span>
						{#if $availableFactors.categorical_factors.includes(key)}
							<span class="badge badge-xs badge-outline ml-auto">cat</span>
						{/if}
					</button>
				{/each}
			{/if}
		{/if}
	</div>
	<div class="border-t border-base-300 pt-2 mt-2">
		<button class="btn btn-xs btn-outline btn-primary w-full" onclick={() => onSelect('custom', 'Custom Factor')}>
			+ Custom Factor
		</button>
	</div>
</div>
