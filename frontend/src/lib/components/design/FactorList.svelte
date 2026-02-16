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

<div class="card bg-base-200 shadow">
	<div class="card-body p-4">
		<h3 class="card-title text-sm">Available Factors</h3>
		<p class="text-xs opacity-60 mb-2">Click to add a factor to your design</p>
		<div class="flex flex-col gap-1 max-h-80 overflow-y-auto">
			{#if $availableFactors}
				{#each Object.entries($availableFactors.factors) as [key, display]}
					<button
						class="btn btn-sm btn-ghost justify-start text-left font-normal"
						onclick={() => onSelect(key, display)}
					>
						<span class="text-xs">{display}</span>
						{#if $availableFactors.categorical_factors.includes(key)}
							<span class="badge badge-xs badge-outline ml-auto">categorical</span>
						{/if}
					</button>
				{/each}
			{/if}
		</div>
		<div class="divider my-1"></div>
		<button class="btn btn-sm btn-outline btn-primary" onclick={() => onSelect('custom', 'Custom Factor')}>
			+ Add Custom Factor
		</button>
	</div>
</div>
