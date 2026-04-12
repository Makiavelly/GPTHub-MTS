<script lang="ts">
	import { onMount } from 'svelte';
	import dayjs from '$lib/dayjs';
	import { toast } from 'svelte-sonner';
	import { deleteMemoryById, getMemories } from '$lib/apis/memories';

	type MemoryItem = {
		id: string;
		content?: string;
		updated_at?: number;
		created_at?: number;
	};

	let memories: MemoryItem[] = [];
	let loading = true;
	let refreshing = false;

	const loadMemories = async () => {
		loading = true;
		try {
			memories = (await getMemories(localStorage.token)) ?? [];
			// Sort newest-updated first
			memories = memories.sort((a, b) => (b.updated_at ?? 0) - (a.updated_at ?? 0));
		} catch (error) {
			memories = [];
			toast.error(`${error}`);
		} finally {
			loading = false;
		}
	};

	const refreshMemories = async () => {
		refreshing = true;
		try {
			memories = (await getMemories(localStorage.token)) ?? [];
			memories = memories.sort((a, b) => (b.updated_at ?? 0) - (a.updated_at ?? 0));
		} catch (error) {
			toast.error(`${error}`);
		} finally {
			refreshing = false;
		}
	};

	const deleteMemory = async (id: string) => {
		try {
			await deleteMemoryById(localStorage.token, id);
			memories = memories.filter((memory) => memory.id !== id);
		} catch (error) {
			toast.error(`${error}`);
		}
	};

	onMount(loadMemories);
</script>

<div class="rounded-3xl border border-gray-200 bg-white/80 p-3 backdrop-blur-sm dark:border-gray-800 dark:bg-gray-900/80">
	<!-- Header -->
	<div class="mb-3 flex items-center justify-between border-b border-gray-200 pb-2 dark:border-gray-800">
		<div class="flex items-center gap-1.5">
			<span class="text-sm font-semibold text-gray-900 dark:text-gray-100">🧠 Долгосрочная память</span>
			<span class="rounded-full bg-gray-100 px-2 py-0.5 text-[11px] text-gray-500 dark:bg-gray-800 dark:text-gray-400">
				{memories.length}
			</span>
		</div>
		<button
			type="button"
			title="Обновить список"
			class="flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] text-gray-500 transition hover:bg-gray-100 hover:text-gray-700 dark:hover:bg-gray-800 dark:hover:text-gray-300 disabled:opacity-40"
			disabled={refreshing}
			on:click={refreshMemories}
		>
			<svg
				class="h-3 w-3 {refreshing ? 'animate-spin' : ''}"
				fill="none"
				stroke="currentColor"
				stroke-width="2"
				viewBox="0 0 24 24"
			>
				<path stroke-linecap="round" stroke-linejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
			</svg>
		</button>
	</div>

	<!-- Auto-extraction hint -->
	<p class="mb-2 text-[10px] leading-4 text-gray-400 dark:text-gray-500">
		Память обновляется автоматически после каждого ответа ИИ
	</p>

	{#if loading}
		<p class="py-4 text-center text-xs text-gray-500 dark:text-gray-400">Загрузка...</p>
	{:else if memories.length === 0}
		<p class="py-4 text-center text-xs leading-6 text-gray-500 dark:text-gray-400">
			Память пуста.<br />
			Заполнится автоматически по мере общения.
		</p>
	{:else}
		<div class="flex max-h-72 flex-col gap-2 overflow-y-auto pr-1">
			{#each memories as memory (memory.id)}
				<div class="group rounded-2xl border border-gray-200 bg-gray-50/80 p-3 dark:border-gray-800 dark:bg-gray-850/70">
					<p class="m-0 text-xs leading-5 text-gray-900 dark:text-gray-100">
						{memory.content ?? ''}
					</p>
					<div class="mt-2 flex items-center justify-between gap-3">
						<span class="text-[10px] text-gray-500 dark:text-gray-400">
							{memory.updated_at ? dayjs(memory.updated_at * 1000).format('DD.MM.YYYY HH:mm') : ''}
						</span>
						<button
							type="button"
							class="text-[10px] text-red-500 opacity-0 transition hover:underline group-hover:opacity-100"
							on:click={() => deleteMemory(memory.id)}
						>
							Удалить
						</button>
					</div>
				</div>
			{/each}
		</div>
	{/if}
</div>
