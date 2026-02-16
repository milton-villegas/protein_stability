<script lang="ts">
	import { designTypes, selectedDesignType } from '$lib/stores/design';
	import { getDesignTypes } from '$lib/api/config';
	import { onMount } from 'svelte';

	interface Props {
		designParams: Record<string, any>;
	}

	let { designParams = $bindable({}) }: Props = $props();

	onMount(async () => {
		if (!$designTypes) {
			$designTypes = await getDesignTypes();
		}
	});

	let currentType = $derived($designTypes?.design_types[$selectedDesignType]);
</script>

<div class="card bg-base-200 shadow">
	<div class="card-body p-4">
		<h3 class="card-title text-sm">Design Type</h3>

		{#if $designTypes}
			<select class="select select-sm select-bordered w-full" bind:value={$selectedDesignType}>
				{#each Object.entries($designTypes.design_types) as [key, info]}
					<option value={key}>{info.display_name}</option>
				{/each}
			</select>

			{#if currentType}
				<p class="text-xs opacity-60 mt-1">{currentType.description}</p>

				{#if currentType.parameters.includes('sample_size') || currentType.parameters.includes('n_samples')}
					<div class="form-control mt-2">
						<label class="label"><span class="label-text text-xs">Sample Size</span></label>
						<input
							type="number"
							class="input input-xs input-bordered"
							min={$designTypes.min_sample_size}
							max={$designTypes.max_sample_size}
							bind:value={designParams.n_samples}
							placeholder="e.g., 20"
						/>
					</div>
				{/if}

				{#if currentType.parameters.includes('resolution')}
					<div class="form-control mt-2">
						<label class="label"><span class="label-text text-xs">Resolution</span></label>
						<select class="select select-xs select-bordered" bind:value={designParams.resolution}>
							{#each $designTypes.resolution_options as opt}
								<option value={opt}>{opt}</option>
							{/each}
						</select>
					</div>
				{/if}

				{#if currentType.parameters.includes('ccd_type')}
					<div class="form-control mt-2">
						<label class="label"><span class="label-text text-xs">CCD Type</span></label>
						<select class="select select-xs select-bordered" bind:value={designParams.ccd_type}>
							{#each $designTypes.ccd_type_options as opt}
								<option value={opt}>{opt}</option>
							{/each}
						</select>
					</div>
				{/if}

				{#if currentType.parameters.includes('model_type')}
					<div class="form-control mt-2">
						<label class="label"><span class="label-text text-xs">Model Type</span></label>
						<select class="select select-xs select-bordered" bind:value={designParams.model_type}>
							{#each $designTypes.d_optimal_model_options as opt}
								<option value={opt}>{opt}</option>
							{/each}
						</select>
					</div>
				{/if}

				{#if currentType.parameters.includes('center_points')}
					<div class="form-control mt-2">
						<label class="label"><span class="label-text text-xs">Center Points</span></label>
						<input type="number" class="input input-xs input-bordered" min="0" max="10" bind:value={designParams.center_points} />
					</div>
				{/if}
			{/if}
		{/if}
	</div>
</div>
