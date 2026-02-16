<script lang="ts">
	import { toast } from '$lib/stores/ui';
</script>

{#if $toast}
	<div class="toast toast-end z-50">
		<div class="alert" class:alert-success={$toast.type === 'success'} class:alert-error={$toast.type === 'error'} class:alert-info={$toast.type === 'info'}>
			<span>{$toast.message}</span>
		</div>
	</div>
{/if}
