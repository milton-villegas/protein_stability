<script lang="ts">
	interface Props {
		summary: Record<string, any>;
	}

	let { summary }: Props = $props();

	let entries = $derived(Object.entries(summary));

	function fmt(n: number, decimals = 4): string {
		if (Math.abs(n) < 0.0001 && n !== 0) return n.toExponential(2);
		return n.toFixed(decimals);
	}

	function confidenceColor(c: string): string {
		if (c === 'HIGH') return 'badge-success';
		if (c === 'MEDIUM') return 'badge-warning';
		return 'badge-error';
	}
</script>

{#each entries as [responseName, data]}
	<div class="card bg-base-200 mb-3">
		<div class="card-body p-4 gap-3">
			<div class="flex items-center gap-2">
				<h4 class="font-bold text-sm">{responseName}</h4>
				<span class="badge badge-sm {confidenceColor(data.confidence)}">
					{data.confidence} confidence
				</span>
				<span class="text-xs opacity-60">
					R²={fmt(data.r_squared)}, {data.n_observations} obs, {data.n_significant} significant factors
				</span>
			</div>

			<!-- Warnings -->
			{#if data.warnings?.length > 0}
				<div class="space-y-1">
					{#each data.warnings as warning}
						<div class="alert alert-warning py-1 px-3 text-xs">
							{warning}
						</div>
					{/each}
				</div>
			{/if}

			<!-- Significant Factors -->
			{#if data.significant_factors?.length > 0}
				<div>
					<h5 class="text-xs font-bold">Significant Factors (p &lt; 0.05) — ranked by effect size</h5>
					<div class="overflow-x-auto">
						<table class="table table-xs mt-1">
							<thead>
								<tr>
									<th class="text-xs">#</th>
									<th class="text-xs">Factor</th>
									<th class="text-xs text-right">Coefficient</th>
									<th class="text-xs text-right">p-value</th>
								</tr>
							</thead>
							<tbody>
								{#each data.significant_factors as sf, i}
									<tr>
										<td class="text-xs">{i + 1}</td>
										<td class="text-xs">{sf.factor}</td>
										<td class="text-xs text-right font-mono">{fmt(sf.coefficient)}</td>
										<td class="text-xs text-right font-mono">{sf.p_value < 0.0001 ? sf.p_value.toExponential(2) : fmt(sf.p_value)}</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				</div>
			{:else}
				<p class="text-xs opacity-60">No significant factors found (p &lt; 0.05)</p>
			{/if}

			<!-- Interactions -->
			{#if data.interactions?.length > 0}
				<div>
					<h5 class="text-xs font-bold">Significant Interactions: {data.interactions.length}</h5>
					{#each data.interactions as inter}
						<p class="text-xs font-mono ml-2">{inter.factor} (coef={fmt(inter.coefficient)}, p={inter.p_value < 0.0001 ? inter.p_value.toExponential(2) : fmt(inter.p_value)})</p>
					{/each}
					<p class="text-xs opacity-70 mt-1">Interactions mean optimal settings depend on factor combinations</p>
				</div>
			{/if}

			<!-- Optimal Direction -->
			{#if data.optimal_directions?.length > 0}
				<div>
					<h5 class="text-xs font-bold">Model-Predicted Optimal Direction</h5>
					<p class="text-xs opacity-60 mb-1">To increase response, adjust these significant factors:</p>
					{#each data.optimal_directions as od}
						<p class="text-xs ml-2">
							<span class="font-mono">{od.direction === 'INCREASE' ? '↑' : '↓'}</span>
							{od.factor}: <span class="font-bold">{od.direction}</span>
							<span class="opacity-60">(effect: {od.effect > 0 ? '+' : ''}{fmt(od.effect)})</span>
						</p>
					{/each}
				</div>
			{/if}

			<!-- Best Experiment -->
			{#if data.best_experiment}
				<div>
					<h5 class="text-xs font-bold">Best Observed Experiment{data.best_experiment.id ? ` (ID: ${data.best_experiment.id})` : ''}</h5>
					<p class="text-xs opacity-60 mb-1">
						{data.best_experiment.direction === 'minimize' ? 'Lowest' : 'Highest'} {responseName}: <span class="font-bold">{fmt(data.best_experiment.value, 2)}</span>
					</p>
					<div class="grid grid-cols-2 gap-x-4 gap-y-0 ml-2">
						{#each Object.entries(data.best_experiment.conditions) as [factor, value]}
							<p class="text-xs"><span class="opacity-60">{factor}:</span> {typeof value === 'number' ? fmt(value, 2) : value}</p>
						{/each}
					</div>
				</div>
			{/if}

			<!-- Next Steps -->
			{#if data.next_steps?.length > 0}
				<details class="mt-1">
					<summary class="text-xs font-bold cursor-pointer">Next Steps</summary>
					<ol class="list-decimal list-inside text-xs mt-1 space-y-0.5 opacity-70">
						{#each data.next_steps as step}
							<li>{step}</li>
						{/each}
					</ol>
				</details>
			{/if}
		</div>
	</div>
{/each}
