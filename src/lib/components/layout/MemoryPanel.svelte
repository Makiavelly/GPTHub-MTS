<script lang="ts">
	import { onMount } from 'svelte';
	import dayjs from '$lib/dayjs';
	import { toast } from 'svelte-sonner';
	import { deleteMemoryById, getMemories } from '$lib/apis/memories';

	type MemoryItem = {
		id: string;
		content?: string;
		scope?: string;
		source_date?: number;
		updated_at?: number;
		created_at?: number;
	};

	const SCOPE_LABELS: Record<string, { label: string; color: string }> = {
		personal:   { label: 'Личное',     color: 'bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300' },
		work:       { label: 'Работа',     color: 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300' },
		preference: { label: 'Предпочт.',  color: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300' },
		general:    { label: 'Общее',      color: 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400' },
	};

	let memories: MemoryItem[] = [];
	let loading = true;
	let refreshing = false;
	let filterScope = 'all';

	$: filteredMemories = filterScope === 'all'
		? memories
		: memories.filter(m => (m.scope ?? 'general') === filterScope);

	const loadMemories = async () => {
		loading = true;
		try {
			const raw = (await getMemories(localStorage.token)) ?? [];
			memories = raw.sort((a: MemoryItem, b: MemoryItem) => (b.updated_at ?? 0) - (a.updated_at ?? 0));
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
			const raw = (await getMemories(localStorage.token)) ?? [];
			memories = raw.sort((a: MemoryItem, b: MemoryItem) => (b.updated_at ?? 0) - (a.updated_at ?? 0));
		} catch (error) {
			toast.error(`${error}`);
		} finally {
			refreshing = false;
		}
	};

	const deleteMemory = async (id: string) => {
		try {
			await deleteMemoryById(localStorage.token, id);
			memories = memories.filter((m) => m.id !== id);
		} catch (error) {
			toast.error(`${error}`);
		}
	};

	onMount(loadMemories);
</script>

<div class="rounded-3xl border border-gray-200 bg-white/80 p-3 backdrop-blur-sm dark:border-gray-800 dark:bg-gray-900/80">
	<!-- Header -->
	<div class="mb-2 flex items-center justify-between border-b border-gray-200 pb-2 dark:border-gray-800">
		<div class="flex items-center gap-1.5">
			<span class="text-sm font-semibold text-gray-900 dark:text-gray-100">🧠 Долгосрочная память</span>
			<span class="rounded-full bg-gray-100 px-2 py-0.5 text-[11px] text-gray-500 dark:bg-gray-800 dark:text-gray-400">
				{filteredMemories.length}
			</span>
		</div>
		<button
			type="button"
			title="Обновить"
			class="rounded-lg p-1 text-gray-400 transition hover:bg-gray-100 hover:text-gray-600 dark:hover:bg-gray-800 disabled:opacity-40"
			disabled={refreshing}
			on:click={refreshMemories}
		>
			<svg class="h-3.5 w-3.5 {refreshing ? 'animate-spin' : ''}" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
				<path stroke-linecap="round" stroke-linejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
			</svg>
		</button>
	</div>

	<!-- Scope filter -->
	<div class="mb-2 flex flex-wrap gap-1">
		{#each ['all', 'personal', 'work', 'preference', 'general'] as scope}
			<button
				type="button"
				class="rounded-full px-2 py-0.5 text-[10px] transition
					{filterScope === scope
						? 'bg-gray-800 text-white dark:bg-gray-200 dark:text-gray-900'
						: 'bg-gray-100 text-gray-500 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-gray-700'}"
				on:click={() => (filterScope = scope)}
			>
				{scope === 'all' ? 'Все' : SCOPE_LABELS[scope]?.label ?? scope}
			</button>
		{/each}
	</div>

	<p class="mb-2 text-[10px] leading-4 text-gray-400 dark:text-gray-500">
		Обновляется автоматически · новые факты сверху
	</p>

	{#if loading}
		<p class="py-4 text-center text-xs text-gray-500 dark:text-gray-400">Загрузка...</p>
	{:else if filteredMemories.length === 0}
		<p class="py-4 text-center text-xs leading-6 text-gray-500 dark:text-gray-400">
			{filterScope === 'all' ? 'Память пуста. Заполнится по мере общения.' : 'Нет фактов в этой категории.'}
		</p>
	{:else}
		<div class="flex max-h-72 flex-col gap-2 overflow-y-auto pr-1">
			{#each filteredMemories as memory (memory.id)}
				{@const scopeInfo = SCOPE_LABELS[memory.scope ?? 'general'] ?? SCOPE_LABELS.general}
				<div class="group rounded-2xl border border-gray-200 bg-gray-50/80 p-3 dark:border-gray-800 dark:bg-gray-850/70">
					<div class="mb-1 flex items-center gap-1.5">
						<span class="rounded-full px-1.5 py-0.5 text-[9px] font-medium {scopeInfo.color}">
							{scopeInfo.label}
						</span>
					</div>
					<p class="m-0 text-xs leading-5 text-gray-900 dark:text-gray-100">
						{memory.content ?? ''}
					</p>
					<div class="mt-1.5 flex items-center justify-between gap-3">
						<span class="text-[10px] text-gray-400 dark:text-gray-500">
							{#if memory.source_date}
								из диалога {dayjs(memory.source_date * 1000).format('DD.MM.YY')} ·
							{/if}
							обновлено {memory.updated_at ? dayjs(memory.updated_at * 1000).format('DD.MM HH:mm') : '—'}
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
