<script lang="ts">
	import { uploadData, responseConfigs } from '$lib/stores/analysis';
	import type { ResponseConfig } from '$lib/types';

	function toggleResponse(name: string) {
		const exists = $responseConfigs.find((r) => r.name === name);
		if (exists) {
			$responseConfigs = $responseConfigs.filter((r) => r.name !== name);
		} else {
			$responseConfigs = [...$responseConfigs, { name, direction: 'maximize' }];
		}
	}

	function updateDirection(name: string, dir: 'maximize' | 'minimize') {
		$responseConfigs = $responseConfigs.map((r) =>
			r.name === name ? { ...r, direction: dir } : r
		);
	}
</script>

{#if $uploadData && $uploadData.potential_responses.length > 0}
	<div class="card bg-base-200 shadow">
		<div class="card-body p-4">
			<h3 class="card-title text-sm">Response Variables</h3>
			<p class="text-xs opacity-60">Select which columns to analyze</p>

			<div class="flex flex-col gap-2 mt-2">
				{#each $uploadData.potential_responses as col}
					{@const config = $responseConfigs.find((r) => r.name === col)}
					<div class="flex items-center gap-3 p-2 rounded bg-base-100">
						<input
							type="checkbox"
							class="checkbox checkbox-sm checkbox-primary"
							checked={!!config}
							onchange={() => toggleResponse(col)}
						/>
						<span class="text-sm flex-1">{col}</span>
						{#if config}
							<select
								class="select select-xs select-bordered"
								value={config.direction}
								onchange={(e) => updateDirection(col, (e.target as HTMLSelectElement).value as any)}
							>
								<option value="maximize">Maximize</option>
								<option value="minimize">Minimize</option>
							</select>
						{/if}
					</div>
				{/each}
			</div>
		</div>
	</div>
{/if}
