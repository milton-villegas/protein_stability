<script lang="ts">
	import { responseConfigs, selectedModelType, useBayesian, analysisResults, plots, modelComparison, suggestions } from '$lib/stores/analysis';
	import { configureAnalysis, runAnalysis, compareModels, getPlot, runOptimization } from '$lib/api/analysis';
	import { showToast } from '$lib/stores/ui';
	import PlotImage from '$lib/components/shared/PlotImage.svelte';
	import DataTable from '$lib/components/shared/DataTable.svelte';

	let running = $state(false);
	let activeTab = $state('results');

	const MODEL_TYPES = [
		{ value: 'linear', label: 'Linear (main effects)' },
		{ value: 'interactions', label: 'Interactions (2-way)' },
		{ value: 'quadratic', label: 'Quadratic (full)' },
		{ value: 'purequadratic', label: 'Pure Quadratic' },
		{ value: 'reduced', label: 'Reduced (backward elimination)' },
		{ value: 'mean', label: 'Mean (intercept only)' },
	];

	async function handleRun() {
		if ($responseConfigs.length === 0) {
			showToast('Select at least one response variable', 'error');
			return;
		}

		running = true;
		try {
			// 1. Configure
			const dirs: Record<string, string> = {};
			$responseConfigs.forEach((r) => (dirs[r.name] = r.direction));

			await configureAnalysis(
				$responseConfigs.map((r) => r.name),
				dirs,
			);

			// 2. Run analysis
			const result = await runAnalysis($selectedModelType);
			$analysisResults = result.results;

			// 3. Load plots
			$plots = {};
			for (const plotType of ['main-effects', 'interactions', 'residuals', 'predictions']) {
				try {
					const p = await getPlot(plotType);
					$plots = { ...$plots, [plotType]: p.image };
				} catch {}
			}

			// 4. Compare models
			try {
				$modelComparison = await compareModels();
			} catch {}

			// 5. Bayesian optimization
			if ($useBayesian) {
				try {
					const opt = await runOptimization(
						$responseConfigs.map((r) => r.name),
						dirs,
					);
					$suggestions = opt.suggestions;
				} catch (e: any) {
					showToast(`Optimization: ${e.message}`, 'error');
				}
			}

			showToast('Analysis complete', 'success');
			activeTab = 'results';
		} catch (e: any) {
			showToast(e.message, 'error');
		}
		running = false;
	}

	let resultEntries = $derived(
		$analysisResults ? Object.entries($analysisResults) : []
	);
</script>

<div class="card bg-base-200 shadow">
	<div class="card-body p-4">
		<h3 class="card-title text-sm">Analysis</h3>

		<div class="flex gap-3 items-end flex-wrap">
			<div class="form-control">
				<label class="label"><span class="label-text text-xs">Model Type</span></label>
				<select class="select select-sm select-bordered" bind:value={$selectedModelType}>
					{#each MODEL_TYPES as m}
						<option value={m.value}>{m.label}</option>
					{/each}
				</select>
			</div>

			<label class="label cursor-pointer gap-2">
				<input type="checkbox" class="toggle toggle-sm toggle-primary" bind:checked={$useBayesian} />
				<span class="label-text text-xs">Bayesian Optimization</span>
			</label>

			<button class="btn btn-sm btn-primary" onclick={handleRun} disabled={running || $responseConfigs.length === 0}>
				{#if running}
					<span class="loading loading-spinner loading-xs"></span>
				{/if}
				Run Analysis
			</button>
		</div>
	</div>
</div>

{#if $analysisResults}
	<div class="mt-4">
		<div role="tablist" class="tabs tabs-bordered">
			<button role="tab" class="tab" class:tab-active={activeTab === 'results'} onclick={() => activeTab = 'results'}>Results</button>
			<button role="tab" class="tab" class:tab-active={activeTab === 'main-effects'} onclick={() => activeTab = 'main-effects'}>Main Effects</button>
			<button role="tab" class="tab" class:tab-active={activeTab === 'interactions'} onclick={() => activeTab = 'interactions'}>Interactions</button>
			<button role="tab" class="tab" class:tab-active={activeTab === 'residuals'} onclick={() => activeTab = 'residuals'}>Residuals</button>
			<button role="tab" class="tab" class:tab-active={activeTab === 'predictions'} onclick={() => activeTab = 'predictions'}>Predictions</button>
			{#if $modelComparison}
				<button role="tab" class="tab" class:tab-active={activeTab === 'comparison'} onclick={() => activeTab = 'comparison'}>Models</button>
			{/if}
			{#if $suggestions}
				<button role="tab" class="tab" class:tab-active={activeTab === 'suggestions'} onclick={() => activeTab = 'suggestions'}>Suggestions</button>
			{/if}
		</div>

		<div class="mt-3">
			{#if activeTab === 'results'}
				{#each resultEntries as [responseName, result]}
					<div class="card bg-base-200 mb-3">
						<div class="card-body p-4">
							<h4 class="font-bold text-sm">{responseName}</h4>

							{#if result.model_stats}
								<div class="stats stats-horizontal shadow text-xs">
									<div class="stat p-2">
										<div class="stat-title text-xs">R-squared</div>
										<div class="stat-value text-sm">{(result.model_stats.r_squared ?? 0).toFixed(4)}</div>
									</div>
									<div class="stat p-2">
										<div class="stat-title text-xs">Adj R-squared</div>
										<div class="stat-value text-sm">{(result.model_stats.adj_r_squared ?? 0).toFixed(4)}</div>
									</div>
									<div class="stat p-2">
										<div class="stat-title text-xs">F-statistic</div>
										<div class="stat-value text-sm">{(result.model_stats.f_statistic ?? 0).toFixed(2)}</div>
									</div>
									<div class="stat p-2">
										<div class="stat-title text-xs">p-value</div>
										<div class="stat-value text-sm">{(result.model_stats.f_pvalue ?? 0).toExponential(2)}</div>
									</div>
								</div>
							{/if}

							{#if result.coefficients}
								<h5 class="text-xs font-bold mt-2">Coefficients</h5>
								{#if Array.isArray(result.coefficients)}
									<DataTable data={result.coefficients} maxRows={20} />
								{/if}
							{/if}
						</div>
					</div>
				{/each}

			{:else if activeTab === 'main-effects' && $plots['main-effects']}
				<PlotImage src={$plots['main-effects']} alt="Main Effects" />

			{:else if activeTab === 'interactions' && $plots['interactions']}
				<PlotImage src={$plots['interactions']} alt="Interactions" />

			{:else if activeTab === 'residuals' && $plots['residuals']}
				<PlotImage src={$plots['residuals']} alt="Residuals" />

			{:else if activeTab === 'predictions' && $plots['predictions']}
				<PlotImage src={$plots['predictions']} alt="Predictions vs Actual" />

			{:else if activeTab === 'comparison' && $modelComparison}
				<div class="card bg-base-200">
					<div class="card-body p-4">
						<h4 class="font-bold text-sm">Model Comparison</h4>
						{#each Object.entries($modelComparison.comparisons ?? {}) as [resp, models]}
							<h5 class="text-xs font-bold mt-2">{resp}</h5>
							{#if $modelComparison.recommendations?.[resp]}
								<span class="badge badge-success badge-sm mb-1">Recommended: {$modelComparison.recommendations[resp]}</span>
							{/if}
							{#if typeof models === 'object' && models !== null}
								<DataTable data={Object.entries(models as Record<string, any>).map(([name, data]: [string, any]) => ({ Model: name, ...data }))} maxRows={10} />
							{/if}
						{/each}
					</div>
				</div>

			{:else if activeTab === 'suggestions' && $suggestions}
				<div class="card bg-base-200">
					<div class="card-body p-4">
						<h4 class="font-bold text-sm">Next Experiment Suggestions (Bayesian Optimization)</h4>
						<DataTable data={$suggestions} />
					</div>
				</div>

			{:else}
				<p class="text-sm text-center opacity-60 p-4">Plot not available</p>
			{/if}
		</div>
	</div>
{/if}
