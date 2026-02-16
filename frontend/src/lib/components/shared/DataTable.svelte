<script lang="ts">
	interface Props {
		data: Record<string, any>[];
		columns?: string[];
		maxRows?: number;
	}

	let { data, columns, maxRows = 100 }: Props = $props();

	let displayColumns = $derived(columns ?? (data.length > 0 ? Object.keys(data[0]) : []));
	let displayData = $derived(data.slice(0, maxRows));
</script>

<div class="overflow-x-auto max-h-96">
	<table class="table table-xs table-pin-rows">
		<thead>
			<tr>
				{#each displayColumns as col}
					<th class="bg-base-200 text-xs">{col}</th>
				{/each}
			</tr>
		</thead>
		<tbody>
			{#each displayData as row}
				<tr class="hover:bg-base-200">
					{#each displayColumns as col}
						<td class="text-xs">{row[col] ?? ''}</td>
					{/each}
				</tr>
			{/each}
		</tbody>
	</table>
	{#if data.length > maxRows}
		<p class="text-xs text-center p-2 opacity-60">Showing {maxRows} of {data.length} rows</p>
	{/if}
</div>
