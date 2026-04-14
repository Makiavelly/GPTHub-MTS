<script lang="ts">
	import { getContext, tick } from 'svelte';
	import { fly } from 'svelte/transition';

	import {
		config,
		user,
		tools as _tools,
		mobile,
		settings,
		toolServers,
		terminalServers
	} from '$lib/stores';

	import { getOAuthClientAuthorizationUrl } from '$lib/apis/configs';
	import { getTools } from '$lib/apis/tools';

	import Knobs from '$lib/components/icons/Knobs.svelte';
	import Check from '$lib/components/icons/Check.svelte';
	import Code from '$lib/components/icons/Code.svelte';
	import Dropdown from '$lib/components/common/Dropdown.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import Wrench from '$lib/components/icons/Wrench.svelte';
	import Sparkles from '$lib/components/icons/Sparkles.svelte';
	import GlobeAlt from '$lib/components/icons/GlobeAlt.svelte';
	import Eye from '$lib/components/icons/Eye.svelte';
	import Photo from '$lib/components/icons/Photo.svelte';
	import Search from '$lib/components/icons/Search.svelte';
	import Terminal from '$lib/components/icons/Terminal.svelte';
	import ChevronRight from '$lib/components/icons/ChevronRight.svelte';
	import ChevronLeft from '$lib/components/icons/ChevronLeft.svelte';

	const i18n = getContext('i18n');

	type ToggleFilter = {
		id: string;
		name: string;
		description?: string;
		icon?: string;
		has_user_valves?: boolean;
	};

	type ToolEntry = {
		name: string;
		description: string;
		enabled: boolean;
		authenticated?: boolean;
		has_user_valves?: boolean;
		serverId?: string;
		authType?: string;
	};

	export let selectedToolIds: string[] = [];

	export let selectedModels: string[] = [];
	export let fileUploadCapableModels: string[] = [];

	export let toggleFilters: ToggleFilter[] = [];
	export let selectedFilterIds: string[] = [];
	export let taskMode = 'auto';
	export let showTaskModes = false;

	export let showWebSearchButton = false;
	export let webSearchEnabled = false;
	export let showImageGenerationButton = false;
	export let imageGenerationEnabled = false;
	export let showCodeInterpreterButton = false;
	export let codeInterpreterEnabled = false;

	export let onShowValves: Function;
	export let onClose: Function;
	export let closeOnOutsideClick = true;
	export let show = false;

	let tab = '';
	let wasOpen = false;

	let tools: Record<string, ToolEntry> = {};
	let toolsReady = false;

	$: if (show && !wasOpen) {
		wasOpen = true;
		init();
	}

	$: if (!show) {
		wasOpen = false;
		tab = '';
	}

	let fileUploadEnabled = true;
	$: fileUploadEnabled =
		fileUploadCapableModels.length === selectedModels.length &&
		($user?.role === 'admin' || $user?.permissions?.chat?.file_upload);

	const init = async () => {
		toolsReady = false;

		if ($_tools === null) {
			await _tools.set(await getTools(localStorage.token));
		}

		const nextTools: Record<string, ToolEntry> = {};
		const availableTools: any[] = Array.isArray($_tools) ? $_tools : [];

		if (availableTools.length > 0) {
			availableTools.forEach((tool: any) => {
				nextTools[tool.id] = {
					name: tool.name,
					description: tool.meta.description,
					enabled: selectedToolIds.includes(tool.id),
					...tool
				};
			});
		}

		if (Array.isArray($toolServers)) {
			for (const serverIdx in $toolServers) {
				const server: any = $toolServers[serverIdx];
				if (server.info) {
					nextTools[`direct_server:${serverIdx}`] = {
						name: server?.info?.title ?? server.url,
						description: server.info.description ?? '',
						enabled: selectedToolIds.includes(`direct_server:${serverIdx}`)
					};
				}
			}
		}

		tools = nextTools;
		selectedToolIds = selectedToolIds.filter((id) => Object.keys(tools).includes(id));
		toolsReady = true;
	};

	const modes = [
		{ id: 'auto', label: 'Авто', icon: Sparkles },
		{ id: 'code', label: 'Код', icon: Code },
		{ id: 'search', label: 'Поиск', icon: Search },
		{ id: 'vision', label: 'Анализ', icon: Eye },
		{ id: 'image', label: 'Картинка', icon: Photo }
	];
</script>

<Dropdown
	bind:show
	{closeOnOutsideClick}
	onOpenChange={(state) => {
		if (state === false) {
			onClose();
		}
	}}
>
	<Tooltip content={$i18n.t('Integrations')} placement="top">
		<slot />
	</Tooltip>
	<div slot="content">
		<div
			class="min-w-70 max-w-70 rounded-2xl px-1 py-1 border border-gray-100 dark:border-gray-800 z-50 bg-white dark:bg-gray-850 dark:text-white shadow-lg max-h-72 overflow-y-auto overflow-x-hidden scrollbar-thin"
		>
			{#if tab === ''}
				<div in:fly={{ x: -20, duration: 150 }}>
					{#if showTaskModes}
						<div class="px-2 pb-1 pt-1">
							<div
								class="px-1 pb-1 text-[11px] font-medium uppercase tracking-[0.18em] text-gray-400 dark:text-gray-500"
							>
								{$i18n.t('Mode')}
							</div>

							<div class="space-y-0.5">
								{#each modes as mode (mode.id)}
									<button
										type="button"
										class="flex w-full items-center justify-between gap-3 rounded-xl px-3 py-2 text-sm transition-colors {taskMode ===
										mode.id
											? 'bg-gray-100 text-gray-900 dark:bg-gray-800 dark:text-gray-100'
											: 'text-gray-600 hover:bg-gray-50 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-800/60 dark:hover:text-gray-100'}"
										aria-pressed={taskMode === mode.id}
										on:click={() => {
											taskMode = mode.id;
											show = false;
										}}
									>
										<div class="flex min-w-0 items-center gap-3">
											<div class="flex size-4 shrink-0 items-center justify-center text-current/80">
												<svelte:component this={mode.icon} className="size-4" strokeWidth="1.6" />
											</div>

											<div class="truncate">{mode.label}</div>
										</div>

										<div class="shrink-0 text-gray-400 dark:text-gray-500">
											{#if taskMode === mode.id}
												<Check className="size-3.5" strokeWidth="1.9" />
											{/if}
										</div>
									</button>
								{/each}
							</div>
						</div>

						<div class="mx-2 my-1 h-px bg-gray-100 dark:bg-gray-800"></div>
					{/if}

					{#if toolsReady}
						{#if Object.keys(tools).length > 0}
							<button
								class="flex w-full items-center justify-between gap-3 rounded-xl px-3 py-2 text-sm text-gray-600 transition-colors hover:bg-gray-50 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-800/60 dark:hover:text-gray-100"
								on:click={() => {
									tab = 'tools';
								}}
							>
								<div class="flex min-w-0 items-center gap-3">
									<div class="flex size-4 shrink-0 items-center justify-center text-current/80">
										<Wrench className="size-4" strokeWidth="1.6" />
									</div>

									<div class="line-clamp-1">
										{$i18n.t('Tools')}
										<span class="ml-0.5 text-gray-500">{Object.keys(tools).length}</span>
									</div>
								</div>

								<div class="shrink-0 text-gray-400 dark:text-gray-500">
									<ChevronRight className="size-4" strokeWidth="1.7" />
								</div>
							</button>
						{/if}
					{:else}
						<div class="py-4">
							<Spinner />
						</div>
					{/if}

					{#if toggleFilters && toggleFilters.length > 0}
						{#each toggleFilters.sort( (a, b) => a.name.localeCompare( b.name, undefined, { sensitivity: 'base' } ) ) as filter, filterIdx (filter.id)}
							<Tooltip content={filter?.description} placement="top-start">
								<button
									class="flex w-full items-center justify-between gap-3 rounded-xl px-3 py-2 text-sm transition-colors {selectedFilterIds.includes(
										filter.id
									)
										? 'bg-gray-100 text-gray-900 dark:bg-gray-800 dark:text-gray-100'
										: 'text-gray-600 hover:bg-gray-50 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-800/60 dark:hover:text-gray-100'}"
									aria-pressed={selectedFilterIds.includes(filter.id)}
									on:click={() => {
										if (selectedFilterIds.includes(filter.id)) {
											selectedFilterIds = selectedFilterIds.filter((id) => id !== filter.id);
										} else {
											selectedFilterIds = [...selectedFilterIds, filter.id];
										}
									}}
								>
									<div class="flex min-w-0 flex-1 items-center gap-3">
										<div class="shrink-0">
											{#if filter?.icon}
												<div class="flex size-4 items-center justify-center opacity-75">
													<img
														src={filter.icon}
														class="size-3.5 {filter.icon.includes('data:image/svg')
															? 'dark:invert-[80%]'
															: ''}"
														style="fill: currentColor;"
														alt={filter.name}
													/>
												</div>
											{:else}
												<Sparkles className="size-4" strokeWidth="1.6" />
											{/if}
										</div>

										<div class="truncate">{filter?.name}</div>
									</div>

									{#if filter?.has_user_valves && ($user?.role === 'admin' || ($user?.permissions?.chat?.valves ?? true))}
										<div class=" shrink-0">
											<Tooltip content={$i18n.t('Valves')}>
												<button
													class="self-center w-fit text-sm text-gray-600 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition rounded-full"
													type="button"
													on:click={(e) => {
														e.stopPropagation();
														e.preventDefault();
														onShowValves({
															type: 'function',
															id: filter.id
														});
													}}
												>
													<Knobs />
												</button>
											</Tooltip>
										</div>
									{/if}

									<div class="shrink-0 text-gray-400 dark:text-gray-500">
										{#if selectedFilterIds.includes(filter.id)}
											<Check className="size-3.5" strokeWidth="1.9" />
										{/if}
									</div>
								</button>
							</Tooltip>
						{/each}
					{/if}

					{#if showWebSearchButton}
						<Tooltip content={$i18n.t('Search the internet')} placement="top-start">
							<button
								class="flex w-full items-center justify-between gap-3 rounded-xl px-3 py-2 text-sm transition-colors {webSearchEnabled
									? 'bg-gray-100 text-gray-900 dark:bg-gray-800 dark:text-gray-100'
									: 'text-gray-600 hover:bg-gray-50 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-800/60 dark:hover:text-gray-100'}"
								aria-pressed={webSearchEnabled}
								on:click={() => {
									webSearchEnabled = !webSearchEnabled;
								}}
							>
								<div class="flex min-w-0 flex-1 items-center gap-3">
									<div class="flex size-4 shrink-0 items-center justify-center text-current/80">
										<GlobeAlt className="size-4" strokeWidth="1.6" />
									</div>

									<div class="truncate">{$i18n.t('Web Search')}</div>
								</div>

								<div class="shrink-0 text-gray-400 dark:text-gray-500">
									{#if webSearchEnabled}
										<Check className="size-3.5" strokeWidth="1.9" />
									{/if}
								</div>
							</button>
						</Tooltip>
					{/if}

					{#if showImageGenerationButton}
						<Tooltip content={$i18n.t('Generate an image')} placement="top-start">
							<button
								class="flex w-full items-center justify-between gap-3 rounded-xl px-3 py-2 text-sm transition-colors {imageGenerationEnabled
									? 'bg-gray-100 text-gray-900 dark:bg-gray-800 dark:text-gray-100'
									: 'text-gray-600 hover:bg-gray-50 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-800/60 dark:hover:text-gray-100'}"
								aria-pressed={imageGenerationEnabled}
								on:click={() => {
									imageGenerationEnabled = !imageGenerationEnabled;
								}}
							>
								<div class="flex min-w-0 flex-1 items-center gap-3">
									<div class="flex size-4 shrink-0 items-center justify-center text-current/80">
										<Photo className="size-4" strokeWidth="1.6" />
									</div>

									<div class="truncate">{$i18n.t('Image')}</div>
								</div>

								<div class="shrink-0 text-gray-400 dark:text-gray-500">
									{#if imageGenerationEnabled}
										<Check className="size-3.5" strokeWidth="1.9" />
									{/if}
								</div>
							</button>
						</Tooltip>
					{/if}

					{#if showCodeInterpreterButton}
						<Tooltip content={$i18n.t('Execute code for analysis')} placement="top-start">
							<button
								class="flex w-full items-center justify-between gap-3 rounded-xl px-3 py-2 text-sm transition-colors {codeInterpreterEnabled
									? 'bg-gray-100 text-gray-900 dark:bg-gray-800 dark:text-gray-100'
									: 'text-gray-600 hover:bg-gray-50 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-800/60 dark:hover:text-gray-100'}"
								aria-pressed={codeInterpreterEnabled}
								aria-label={codeInterpreterEnabled
									? $i18n.t('Disable Code Interpreter')
									: $i18n.t('Enable Code Interpreter')}
								on:click={() => {
									codeInterpreterEnabled = !codeInterpreterEnabled;
								}}
							>
								<div class="flex min-w-0 flex-1 items-center gap-3">
									<div class="flex size-4 shrink-0 items-center justify-center text-current/80">
										<Terminal className="size-4" strokeWidth="1.6" />
									</div>

									<div class="truncate">{$i18n.t('Code Interpreter')}</div>
								</div>

								<div class="shrink-0 text-gray-400 dark:text-gray-500">
									{#if codeInterpreterEnabled}
										<Check className="size-3.5" strokeWidth="1.9" />
									{/if}
								</div>
							</button>
						</Tooltip>
					{/if}
				</div>
			{:else if tab === 'tools' && toolsReady}
				<div in:fly={{ x: 20, duration: 150 }}>
					<button
						class="flex w-full items-center justify-between gap-3 rounded-xl px-3 py-2 text-sm text-gray-600 transition-colors hover:bg-gray-50 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-800/60 dark:hover:text-gray-100"
						on:click={() => {
							tab = '';
						}}
					>
						<div class="flex min-w-0 items-center gap-3">
							<div class="flex size-4 shrink-0 items-center justify-center text-current/80">
								<ChevronLeft className="size-4" strokeWidth="1.7" />
							</div>

							<div>
								{$i18n.t('Tools')}
								<span class="ml-0.5 text-gray-500">{Object.keys(tools).length}</span>
							</div>
						</div>
					</button>

					{#each Object.keys(tools) as toolId}
						<button
							class="relative flex w-full items-center justify-between gap-3 rounded-xl px-3 py-2 text-sm transition-colors {tools[
								toolId
							].enabled
								? 'bg-gray-100 text-gray-900 dark:bg-gray-800 dark:text-gray-100'
								: 'text-gray-600 hover:bg-gray-50 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-800/60 dark:hover:text-gray-100'}"
							aria-pressed={tools[toolId].enabled}
							on:click={async (e) => {
								if (!(tools[toolId]?.authenticated ?? true)) {
									e.preventDefault();

									let parts = toolId.split(':');
									let serverId = parts?.at(-1) ?? toolId;

									// Persist the tool ID so we can re-enable it after OAuth redirect
									sessionStorage.setItem('pendingOAuthToolId', toolId);

									const authUrl = getOAuthClientAuthorizationUrl(serverId, 'mcp');
									window.open(authUrl, '_self', 'noopener');
								} else {
									tools[toolId].enabled = !tools[toolId].enabled;

									const state = tools[toolId].enabled;
									await tick();

									if (state) {
										selectedToolIds = [...selectedToolIds, toolId];
									} else {
										selectedToolIds = selectedToolIds.filter((id) => id !== toolId);
									}
								}
							}}
						>
							{#if !(tools[toolId]?.authenticated ?? true)}
								<!-- make it slighly darker and not clickable -->
								<div class="absolute inset-0 opacity-50 rounded-xl cursor-pointer z-10"></div>
							{/if}
							<div class="flex min-w-0 flex-1 items-center gap-3">
								<div class="flex min-w-0 flex-1 items-center gap-3">
									<Tooltip content={tools[toolId]?.name ?? ''} placement="top">
										<div class="flex size-4 shrink-0 items-center justify-center text-current/80">
											<Wrench className="size-4" strokeWidth="1.6" />
										</div>
									</Tooltip>
									<Tooltip content={tools[toolId]?.description ?? ''} placement="top-start">
										<div class="truncate">{tools[toolId].name}</div>
									</Tooltip>
								</div>
							</div>

							{#if tools[toolId]?.has_user_valves && ($user?.role === 'admin' || ($user?.permissions?.chat?.valves ?? true))}
								<div class=" shrink-0">
									<Tooltip content={$i18n.t('Valves')}>
										<button
											class="self-center w-fit text-sm text-gray-600 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition rounded-full"
											type="button"
											on:click={(e) => {
												e.stopPropagation();
												e.preventDefault();
												onShowValves({
													type: 'tool',
													id: toolId
												});
											}}
										>
											<Knobs />
										</button>
									</Tooltip>
								</div>
							{/if}

							<div class="shrink-0 text-gray-400 dark:text-gray-500">
								{#if tools[toolId].enabled}
									<Check className="size-3.5" strokeWidth="1.9" />
								{/if}
							</div>
						</button>
					{/each}
				</div>
			{/if}
		</div>
	</div>
</Dropdown>
