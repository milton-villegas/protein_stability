<script lang="ts">
	interface Props {
		src: string;
		alt?: string;
	}

	let { src, alt = 'Plot' }: Props = $props();
</script>

<div class="p-2">
	<img {src} {alt} class="w-full max-w-3xl mx-auto rounded-lg shadow" />
</div>
