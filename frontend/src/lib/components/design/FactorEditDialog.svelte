<script lang="ts">
	import { availableFactors } from '$lib/stores/design';
	import { getConstraints } from '$lib/api/config';
	import { onMount } from 'svelte';

	interface Props {
		open: boolean;
		factorName: string;
		isNew: boolean;
		initialLevels?: string[];
		initialStockConc?: number | null;
		onSave: (name: string, levels: string[], stockConc: number | null, perLevelConcs: any) => void;
		onClose: () => void;
	}

	let {
		open,
		factorName,
		isNew,
		initialLevels = [],
		initialStockConc = null,
		onSave,
		onClose,
	}: Props = $props();

	let name = $state(factorName);
	let levels = $state<string[]>([...initialLevels]);
	let stockConc = $state<string>(initialStockConc?.toString() ?? '');
	let newLevel = $state('');
	let seqFrom = $state('');
	let seqTo = $state('');
	let seqStep = $state('');
	let validationError = $state('');
	let constraints: Record<string, any> = $state({});

	let isCategorical = $derived(
		$availableFactors?.categorical_factors.includes(name) ?? false
	);

	let stockWarning = $derived(() => {
		if (isCategorical || !stockConc) return '';
		const sc = parseFloat(stockConc);
		if (isNaN(sc)) return '';
		for (const lv of levels) {
			const num = parseFloat(lv);
			if (!isNaN(num) && num > sc) {
				return `Level ${lv} exceeds stock concentration (${stockConc})`;
			}
		}
		return '';
	});

	onMount(async () => {
		try {
			const data = await getConstraints();
			constraints = data.constraints ?? {};
		} catch {}
	});

	// Reset form when dialog opens
	$effect(() => {
		if (open) {
			name = factorName;
			levels = [...initialLevels];
			stockConc = initialStockConc?.toString() ?? '';
			newLevel = '';
			validationError = '';
		}
	});

	function validateLevel(val: string, factorKey: string): string | null {
		if (isCategorical) return null; // No numeric validation for categorical

		const num = parseFloat(val);
		if (isNaN(num)) return null; // Non-numeric levels accepted (could be categorical)

		const constraint = constraints[factorKey];
		if (!constraint) return null;

		if (constraint.min !== undefined && num < constraint.min) {
			return `${val} is below minimum (${constraint.min}). ${constraint.description ?? ''}`;
		}
		if (constraint.max !== undefined && num > constraint.max) {
			return `${val} exceeds maximum (${constraint.max}). ${constraint.description ?? ''}`;
		}
		return null;
	}

	function addLevel() {
		const val = newLevel.trim();
		if (!val) return;

		// Support comma-separated values
		const values = val.includes(',') ? val.split(',').map(v => v.trim()).filter(Boolean) : [val];

		for (const v of values) {
			const error = validateLevel(v, name);
			if (error) {
				validationError = error;
				return;
			}
			if (!levels.includes(v)) {
				levels = [...levels, v];
			}
		}
		validationError = '';
		newLevel = '';
	}

	function removeLevel(idx: number) {
		levels = levels.filter((_, i) => i !== idx);
	}

	function generateSequence() {
		const from = parseFloat(seqFrom);
		const to = parseFloat(seqTo);
		const step = parseFloat(seqStep);
		if (isNaN(from) || isNaN(to) || isNaN(step) || step <= 0) return;

		// Validate range
		const constraint = constraints[name];
		if (constraint) {
			if (constraint.min !== undefined && from < constraint.min) {
				validationError = `From value ${from} is below minimum (${constraint.min})`;
				return;
			}
			if (constraint.max !== undefined && to > constraint.max) {
				validationError = `To value ${to} exceeds maximum (${constraint.max})`;
				return;
			}
		}

		const newLevels: string[] = [];
		for (let v = from; v <= to + step / 100; v += step) {
			const rounded = Math.round(v * 1000) / 1000;
			newLevels.push(rounded.toString());
		}
		levels = newLevels;
		validationError = '';
		seqFrom = '';
		seqTo = '';
		seqStep = '';
	}

	function handleSave() {
		if (!name.trim() || levels.length === 0) return;
		const sc = stockConc ? parseFloat(stockConc) : null;
		onSave(name.trim(), levels, isNaN(sc as any) ? null : sc, null);
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			e.preventDefault();
			addLevel();
		}
	}
</script>

{#if open}
	<dialog class="modal modal-open">
		<div class="modal-box max-w-lg">
			<h3 class="text-lg font-bold mb-4">{isNew ? 'Add Factor' : `Edit: ${factorName}`}</h3>

			{#if isNew && factorName === 'custom'}
				<div class="form-control mb-3">
					<label class="label"><span class="label-text text-sm">Factor Name</span></label>
					<input type="text" class="input input-sm input-bordered" bind:value={name} placeholder="e.g., KCl" />
				</div>
			{/if}

			{#if !isCategorical}
				<div class="form-control mb-3">
					<label class="label"><span class="label-text text-sm">Stock Concentration</span></label>
					<input type="number" class="input input-sm input-bordered" bind:value={stockConc} placeholder="e.g., 5000" step="any" />
					{#if stockWarning()}
						<label class="label"><span class="label-text-alt text-warning text-xs">{stockWarning()}</span></label>
					{/if}
				</div>
			{/if}

			{#if constraints[name]}
				<p class="text-xs opacity-60 mb-2">Range: {constraints[name].description}</p>
			{/if}

			<div class="form-control mb-3">
				<label class="label"><span class="label-text text-sm">Levels ({levels.length})</span></label>
				<div class="flex gap-2">
					<input
						type="text"
						class="input input-sm input-bordered flex-1"
						bind:value={newLevel}
						placeholder={isCategorical ? 'e.g., Tween-20' : 'e.g., 100 or 50,100,200'}
						onkeydown={handleKeydown}
					/>
					<button class="btn btn-sm btn-primary" onclick={addLevel}>Add</button>
				</div>
				{#if validationError}
					<label class="label"><span class="label-text-alt text-error text-xs">{validationError}</span></label>
				{/if}
			</div>

			{#if levels.length > 0}
				<div class="flex flex-wrap gap-1 mb-3 max-h-32 overflow-y-auto p-2 bg-base-200 rounded">
					{#each levels as level, idx}
						<span class="badge badge-outline gap-1">
							{level}
							<button class="text-error text-xs" onclick={() => removeLevel(idx)}>x</button>
						</span>
					{/each}
				</div>
			{/if}

			{#if !isCategorical}
				<div class="collapse collapse-arrow bg-base-200 mb-3">
					<input type="checkbox" />
					<div class="collapse-title text-sm font-medium">Generate Sequence</div>
					<div class="collapse-content">
						<div class="grid grid-cols-3 gap-2">
							<div class="form-control">
								<label class="label"><span class="label-text text-xs">From</span></label>
								<input type="number" class="input input-xs input-bordered" bind:value={seqFrom} step="any" />
							</div>
							<div class="form-control">
								<label class="label"><span class="label-text text-xs">To</span></label>
								<input type="number" class="input input-xs input-bordered" bind:value={seqTo} step="any" />
							</div>
							<div class="form-control">
								<label class="label"><span class="label-text text-xs">Step</span></label>
								<input type="number" class="input input-xs input-bordered" bind:value={seqStep} step="any" />
							</div>
						</div>
						<button class="btn btn-xs btn-outline mt-2 w-full" onclick={generateSequence}>Generate</button>
					</div>
				</div>
			{/if}

			<div class="modal-action">
				<button class="btn btn-sm" onclick={onClose}>Cancel</button>
				<button class="btn btn-sm btn-primary" onclick={handleSave} disabled={!name.trim() || levels.length === 0}>Save</button>
			</div>
		</div>
		<form method="dialog" class="modal-backdrop">
			<button onclick={onClose}>close</button>
		</form>
	</dialog>
{/if}
