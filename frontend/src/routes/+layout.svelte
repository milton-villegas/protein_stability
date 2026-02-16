<script lang="ts">
	import '../app.css';
	import Header from '$lib/components/layout/Header.svelte';
	import Toast from '$lib/components/layout/Toast.svelte';
	import { sessionActive, projectInfo } from '$lib/stores/project';
	import { createProject } from '$lib/api/project';
	import { onMount } from 'svelte';
	import { page } from '$app/state';

	let { children } = $props();

	onMount(async () => {
		if (!$sessionActive) {
			try {
				await createProject('Untitled Project');
				sessionActive.set(true);
				projectInfo.set({ name: 'Untitled Project', has_design: false, has_results: false, factors_count: 0, design_runs: null });
			} catch {}
		}
	});

	const tabs = [
		{ label: 'Design', href: '/design' },
		{ label: 'Analysis', href: '/analysis' },
	];
</script>

<div class="min-h-screen bg-base-100 flex flex-col">
	<Header />

	<div class="flex justify-center px-4 pt-3">
		<div role="tablist" class="tabs tabs-boxed">
			{#each tabs as tab}
				<a
					role="tab"
					href={tab.href}
					class="tab"
					class:tab-active={page.url.pathname.startsWith(tab.href)}
				>
					{tab.label}
				</a>
			{/each}
		</div>
	</div>

	<main class="flex-1 p-4 max-w-7xl mx-auto w-full">
		{@render children()}
	</main>

	<footer class="text-center text-xs opacity-40 py-2">
		SCOUT v2.0.0
	</footer>

	<Toast />
</div>
