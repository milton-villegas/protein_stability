<script lang="ts">
	interface Props {
		data: Record<string, any>[];
		columns?: string[];
		maxRows?: number;
	}

	let { data, columns, maxRows = 100 }: Props = $props();

	let displayColumns = $derived(columns ?? (data.length > 0 ? Object.keys(data[0]) : []));
	let displayData = $derived(data.slice(0, maxRows));

	function formatCell(value: any): string {
		if (value === null || value === undefined || value === '') return '';
		if (typeof value === 'boolean') return value ? 'Yes' : 'No';
		if (typeof value === 'number') {
			if (Number.isInteger(value)) return value.toString();
			const abs = Math.abs(value);
			if (abs === 0) return '0';
			if (abs < 0.0001) return value.toExponential(2);
			return value.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
		}
		return String(value);
	}
</script>

<div class="overflow-auto h-full border border-base-300 rounded-lg">
	<table class="table table-xs table-pin-rows table-pin-cols">
		<thead>
			<tr>
				{#each displayColumns as col, i}
					{#if i === 0}
						<th class="bg-base-300 text-xs font-semibold z-20">{col}</th>
					{:else}
						<th class="bg-base-200 text-xs font-semibold whitespace-nowrap">{col}</th>
					{/if}
				{/each}
			</tr>
		</thead>
		<tbody>
			{#each displayData as row}
				<tr class="hover:bg-base-200">
					{#each displayColumns as col, i}
						{#if i === 0}
							<th class="bg-base-100 text-xs font-normal z-10">{formatCell(row[col])}</th>
						{:else}
							<td class="text-xs whitespace-nowrap">{formatCell(row[col])}</td>
						{/if}
					{/each}
				</tr>
			{/each}
		</tbody>
	</table>
</div>
{#if data.length > maxRows}
	<p class="text-xs text-center p-1 opacity-60">Showing {maxRows} of {data.length} rows</p>
{/if}
