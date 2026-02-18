<script lang="ts">
	import { responseConfigs, selectedModelType, useBayesian, analysisResults, plots, modelComparison, suggestions, analysisSummary, hasPareto, paretoPoints } from '$lib/stores/analysis';
	import { configureAnalysis, runAnalysis, compareModels, getPlot, runOptimization, getAnalysisSummary, exportResults, exportBOBatch, exportBOCsv, getParetoFrontier } from '$lib/api/analysis';
	import { showToast } from '$lib/stores/ui';
	import PlotImage from '$lib/components/shared/PlotImage.svelte';
	import DataTable from '$lib/components/shared/DataTable.svelte';
	import ResultsDetail from '$lib/components/analysis/ResultsDetail.svelte';

	let running = $state(false);
	let exporting = $state(false);
	let activeTab = $state('results');
	let selectedPlotResponse = $state('');
	let loadingPlots = $state(false);
	let nSuggestions = $state(5);
	let explorationMode = $state(false);
	let batchNumber = $state(1);
	let boFinalVolume = $state(100);
	let boPlots = $state<Record<string, string>>({});

	const MODEL_TYPES = [
		{ value: 'auto', label: 'Auto (Recommended)' },
		{ value: 'linear', label: 'Linear (main effects)' },
		{ value: 'interactions', label: 'Interactions (2-way)' },
		{ value: 'quadratic', label: 'Quadratic (full)' },
		{ value: 'purequadratic', label: 'Pure Quadratic' },
		{ value: 'reduced', label: 'Reduced (backward elimination)' },
		{ value: 'mean', label: 'Mean (intercept only)' },
	];

	const PLOT_TYPES = ['main-effects', 'interactions', 'residuals', 'predictions', 'distribution', 'qq'];

	async function loadPlots(response?: string) {
		loadingPlots = true;
		$plots = {};
		for (const plotType of PLOT_TYPES) {
			try {
				const p = await getPlot(plotType, response);
				$plots = { ...$plots, [plotType]: p.image };
			} catch {}
		}
		loadingPlots = false;
	}

	async function handlePlotResponseChange(newResponse: string) {
		selectedPlotResponse = newResponse;
		await loadPlots(newResponse || undefined);
	}

	async function handleRun() {
		if ($responseConfigs.length === 0) {
			showToast('Select at least one response variable', 'error');
			return;
		}

		running = true;
		try {
			// 1. Configure
			const dirs: Record<string, string> = {};
			const cons: Record<string, Record<string, number>> = {};
			$responseConfigs.forEach((r) => {
				dirs[r.name] = r.direction;
				const c: Record<string, number> = {};
				if (r.min != null) c.min = r.min;
				if (r.max != null) c.max = r.max;
				if (Object.keys(c).length > 0) cons[r.name] = c;
			});

			await configureAnalysis(
				$responseConfigs.map((r) => r.name),
				dirs,
				Object.keys(cons).length > 0 ? cons : undefined,
			);

			// 2. Compare models first (needed for auto mode and Models tab)
			$modelComparison = null;
			let bestModel = $selectedModelType;
			try {
				$modelComparison = await compareModels();
				if ($selectedModelType === 'auto' && $modelComparison?.recommendations) {
					const firstResp = Object.keys($modelComparison.recommendations)[0];
					if (firstResp) {
						bestModel = $modelComparison.recommendations[firstResp].best_model || 'linear';
					}
				}
			} catch {}

			// 3. Run analysis with selected (or auto-detected) model
			const result = await runAnalysis(bestModel);
			$analysisResults = result.results;

			// 4. Load plots (for first response)
			selectedPlotResponse = '';
			await loadPlots();

			// 5. Get analysis summary
			try {
				$analysisSummary = await getAnalysisSummary();
			} catch {}

			// 6. Bayesian optimization
			$suggestions = null;
			$hasPareto = false;
			$paretoPoints = null;
			boPlots = {};
			if ($useBayesian) {
				try {
					const opt = await runOptimization(
						$responseConfigs.map((r) => r.name),
						dirs,
						Object.keys(cons).length > 0 ? cons : undefined,
						nSuggestions,
						explorationMode,
					);
					$suggestions = opt.suggestions;
					$hasPareto = opt.has_pareto ?? false;

					// Fetch Pareto frontier if available
					if ($hasPareto) {
						try {
							const pareto = await getParetoFrontier();
							$paretoPoints = pareto.pareto_points;
						} catch {}
					}

					// Fetch BO-specific plots
					const isMultiObjective = $responseConfigs.length > 1;
					if (isMultiObjective) {
						try {
							const p = await getPlot('bo-pareto');
							boPlots = { ...boPlots, 'bo-pareto': p.image };
						} catch {}
						try {
							const p = await getPlot('bo-parallel');
							boPlots = { ...boPlots, 'bo-parallel': p.image };
						} catch {}
					} else {
						try {
							const p = await getPlot('bo-response-surface');
							boPlots = { ...boPlots, 'bo-response-surface': p.image };
						} catch {}
					}
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

	async function handleExportResults() {
		exporting = true;
		try {
			await exportResults();
			showToast('Results exported', 'success');
		} catch (e: any) {
			showToast(`Export failed: ${e.message}`, 'error');
		}
		exporting = false;
	}

	async function handleExportBO() {
		exporting = true;
		try {
			await exportBOBatch(boFinalVolume, batchNumber);
			showToast('BO batch exported (Excel)', 'success');
		} catch (e: any) {
			showToast(`Export failed: ${e.message}`, 'error');
		}
		exporting = false;
	}

	async function handleExportBOCsv() {
		exporting = true;
		try {
			await exportBOCsv(boFinalVolume, batchNumber);
			showToast('BO batch exported (CSV)', 'success');
		} catch (e: any) {
			showToast(`Export failed: ${e.message}`, 'error');
		}
		exporting = false;
	}

	let resultEntries = $derived(
		$analysisResults ? Object.entries($analysisResults) : []
	);

	let isPlotTab = $derived(
		['main-effects', 'interactions', 'residuals', 'predictions', 'distribution', 'qq'].includes(activeTab)
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

			{#if $useBayesian}
				<div class="form-control">
					<label class="label"><span class="label-text text-xs">Suggestions</span></label>
					<input type="number" class="input input-sm input-bordered w-20" min="1" max="20" bind:value={nSuggestions} />
				</div>
				<label class="label cursor-pointer gap-2">
					<input type="checkbox" class="toggle toggle-sm toggle-secondary" bind:checked={explorationMode} />
					<span class="label-text text-xs">Exploration Mode</span>
				</label>
			{/if}

			<button class="btn btn-sm btn-primary" onclick={handleRun} disabled={running || $responseConfigs.length === 0}>
				{#if running}
					<span class="loading loading-spinner loading-xs"></span>
				{/if}
				Run Analysis
			</button>

			{#if $analysisResults}
				<button class="btn btn-sm btn-outline" onclick={handleExportResults} disabled={exporting}>
					Export Statistics
				</button>
			{/if}
			{#if $suggestions}
				<button class="btn btn-sm btn-outline" onclick={handleExportBO} disabled={exporting}>
					Export BO Batch
				</button>
			{/if}
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
			<button role="tab" class="tab" class:tab-active={activeTab === 'distribution'} onclick={() => activeTab = 'distribution'}>Distribution</button>
			<button role="tab" class="tab" class:tab-active={activeTab === 'qq'} onclick={() => activeTab = 'qq'}>Q-Q Plot</button>
			{#if $modelComparison}
				<button role="tab" class="tab" class:tab-active={activeTab === 'comparison'} onclick={() => activeTab = 'comparison'}>Models</button>
			{/if}
			{#if $suggestions}
				<button role="tab" class="tab" class:tab-active={activeTab === 'suggestions'} onclick={() => activeTab = 'suggestions'}>Suggestions</button>
			{/if}
			{#if $suggestions && Object.keys(boPlots).length > 0}
				<button role="tab" class="tab" class:tab-active={activeTab === 'bo-plots'} onclick={() => activeTab = 'bo-plots'}>BO Plots</button>
			{/if}
		</div>

		<!-- Response selector for plot tabs (multi-response) -->
		{#if isPlotTab && resultEntries.length > 1}
			<div class="flex items-center gap-2 mt-2">
				<span class="text-xs opacity-60">Response:</span>
				<select
					class="select select-xs select-bordered"
					value={selectedPlotResponse}
					onchange={(e) => handlePlotResponseChange((e.target as HTMLSelectElement).value)}
				>
					<option value="">All</option>
					{#each resultEntries as [name]}
						<option value={name}>{name}</option>
					{/each}
				</select>
				{#if loadingPlots}
					<span class="loading loading-spinner loading-xs"></span>
				{/if}
			</div>
		{/if}

		<!-- Tab panels: use display to toggle instead of {#if} to prevent scroll jumps -->
		<div class="mt-3" style:display={activeTab === 'results' ? 'block' : 'none'}>
			{#if $analysisSummary}
				<ResultsDetail summary={$analysisSummary} />
			{/if}

			{#each resultEntries as [responseName, result]}
				<details class="collapse collapse-arrow bg-base-200 mb-3">
					<summary class="collapse-title text-sm font-bold p-4 min-h-0">
						{responseName} — Model Statistics
					</summary>
					<div class="collapse-content px-4 pb-4">
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
				</details>
			{/each}
		</div>

		<div class="mt-3" style:display={activeTab === 'main-effects' ? 'block' : 'none'}>
			{#if $plots['main-effects']}
				<PlotImage src={$plots['main-effects']} alt="Main Effects" />
			{:else}
				<p class="text-sm text-center opacity-60 p-4">Plot not available</p>
			{/if}
		</div>

		<div class="mt-3" style:display={activeTab === 'interactions' ? 'block' : 'none'}>
			{#if $plots['interactions']}
				<PlotImage src={$plots['interactions']} alt="Interactions" />
			{:else}
				<p class="text-sm text-center opacity-60 p-4">Plot not available</p>
			{/if}
		</div>

		<div class="mt-3" style:display={activeTab === 'residuals' ? 'block' : 'none'}>
			{#if $plots['residuals']}
				<PlotImage src={$plots['residuals']} alt="Residuals" />
			{:else}
				<p class="text-sm text-center opacity-60 p-4">Plot not available</p>
			{/if}
		</div>

		<div class="mt-3" style:display={activeTab === 'predictions' ? 'block' : 'none'}>
			{#if $plots['predictions']}
				<PlotImage src={$plots['predictions']} alt="Predictions vs Actual" />
			{:else}
				<p class="text-sm text-center opacity-60 p-4">Plot not available</p>
			{/if}
		</div>

		<div class="mt-3" style:display={activeTab === 'distribution' ? 'block' : 'none'}>
			{#if $plots['distribution']}
				<PlotImage src={$plots['distribution']} alt="Response Distribution" />
			{:else}
				<p class="text-sm text-center opacity-60 p-4">Plot not available</p>
			{/if}
		</div>

		<div class="mt-3" style:display={activeTab === 'qq' ? 'block' : 'none'}>
			{#if $plots['qq']}
				<PlotImage src={$plots['qq']} alt="Q-Q Plot" />
			{:else}
				<p class="text-sm text-center opacity-60 p-4">Plot not available</p>
			{/if}
		</div>

		{#if $modelComparison}
			<div class="mt-3" style:display={activeTab === 'comparison' ? 'block' : 'none'}>
				<div class="card bg-base-200">
					<div class="card-body p-4">
						<h4 class="font-bold text-sm">Model Comparison</h4>
						<p class="text-xs opacity-60">Selection criteria: Adjusted R² (60%), BIC (30%), Parsimony (10%). Higher Adj R² is better | Lower BIC is better | Simpler models preferred.</p>
						{#each Object.entries($modelComparison.comparisons ?? {}) as [resp, models]}
							<h5 class="text-xs font-bold mt-2">{resp}</h5>
							{@const rec = $modelComparison.recommendations?.[resp]}
							{#if rec}
								<div class="mb-2">
									<span class="badge badge-success badge-sm">Recommended: {typeof rec === 'string' ? rec : rec.best_model}</span>
									{#if typeof rec === 'object' && rec.reason}
										<p class="text-xs opacity-70 mt-1">{rec.reason}</p>
									{/if}
								</div>
							{/if}
							{#if typeof models === 'object' && models !== null}
								<DataTable data={Object.entries(models as Record<string, any>).map(([name, data]: [string, any]) => ({ Model: name, ...data }))} maxRows={10} />
							{/if}
							{#if typeof rec === 'object' && rec.scores}
								<details class="mt-2">
									<summary class="text-xs cursor-pointer opacity-60">Model Scoring Details</summary>
									<DataTable data={Object.entries(rec.scores as Record<string, any>).map(([name, data]: [string, any]) => ({ Model: name, ...data }))} maxRows={10} />
								</details>
							{/if}
						{/each}
					</div>
				</div>
			</div>
		{/if}

		{#if $suggestions}
			<div class="mt-3" style:display={activeTab === 'suggestions' ? 'block' : 'none'}>
				<div class="card bg-base-200">
					<div class="card-body p-4">
						<div class="flex items-center justify-between flex-wrap gap-2">
							<h4 class="font-bold text-sm">Next Experiment Suggestions (Bayesian Optimization)</h4>
							<div class="flex items-end gap-2 flex-wrap">
								<div class="form-control">
									<label class="label py-0"><span class="label-text text-xs">Batch #</span></label>
									<input type="number" class="input input-xs input-bordered w-16" min="1" bind:value={batchNumber} />
								</div>
								<div class="form-control">
									<label class="label py-0"><span class="label-text text-xs">Volume (µL)</span></label>
									<input type="number" class="input input-xs input-bordered w-20" min="1" bind:value={boFinalVolume} />
								</div>
								<button class="btn btn-sm btn-outline" onclick={handleExportBO} disabled={exporting}>
									Export Excel
								</button>
								<button class="btn btn-sm btn-outline" onclick={handleExportBOCsv} disabled={exporting}>
									Export CSV
								</button>
							</div>
						</div>
						<p class="text-xs opacity-60">These conditions are predicted to improve your response(s). Run these experiments and re-analyze to refine the model.</p>
						<DataTable data={$suggestions} />

						{#if $hasPareto && $paretoPoints && $paretoPoints.length > 0}
							<div class="divider my-2"></div>
							<h4 class="font-bold text-sm">Pareto Frontier</h4>
							<p class="text-xs opacity-60">Pareto-optimal points from your existing data — no single objective can be improved without worsening another.</p>
							<DataTable data={$paretoPoints.map((p: any) => ({ ...p.parameters, ...p.objectives }))} />
						{/if}
					</div>
				</div>
			</div>
		{/if}

		{#if $suggestions && Object.keys(boPlots).length > 0}
			<div class="mt-3" style:display={activeTab === 'bo-plots' ? 'block' : 'none'}>
				<div class="space-y-4">
					{#if boPlots['bo-response-surface']}
						<div>
							<h4 class="font-bold text-sm mb-2">Response Surface</h4>
							<PlotImage src={boPlots['bo-response-surface']} alt="BO Response Surface" />
						</div>
					{/if}
					{#if boPlots['bo-pareto']}
						<div>
							<h4 class="font-bold text-sm mb-2">Pareto Frontier</h4>
							<PlotImage src={boPlots['bo-pareto']} alt="Pareto Frontier" />
						</div>
					{/if}
					{#if boPlots['bo-parallel']}
						<div>
							<h4 class="font-bold text-sm mb-2">Parallel Coordinates</h4>
							<PlotImage src={boPlots['bo-parallel']} alt="Parallel Coordinates" />
						</div>
					{/if}
				</div>
			</div>
		{/if}
	</div>
{/if}
