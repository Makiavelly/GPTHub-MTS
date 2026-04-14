<script lang="ts">
	import { marked } from 'marked';

	import { getContext, tick } from 'svelte';
	import dayjs from '$lib/dayjs';

	import { mobile, settings, user } from '$lib/stores';
	import { WEBUI_API_BASE_URL, WEBUI_BASE_URL } from '$lib/constants';

	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import { copyToClipboard, sanitizeResponseContent } from '$lib/utils';
	import ArrowUpTray from '$lib/components/icons/ArrowUpTray.svelte';
	import Check from '$lib/components/icons/Check.svelte';
	import ModelItemMenu from './ModelItemMenu.svelte';
	import EllipsisHorizontal from '$lib/components/icons/EllipsisHorizontal.svelte';
	import { toast } from 'svelte-sonner';
	import Tag from '$lib/components/icons/Tag.svelte';
	import Label from '$lib/components/icons/Label.svelte';

	const i18n = getContext('i18n');

	export let selectedModelIdx: number = -1;
	export let item: any = {};
	export let index: number = -1;
	export let value: string = '';

	export let unloadModelHandler: (modelValue: string) => void = () => {};
	export let pinModelHandler: (modelId: string) => void = () => {};

	export let onClick: () => void = () => {};
	let itemCardInfo;

	const modelInfo = {
		'mws-gpt-alpha': {
			icon: '✨',
			desc: 'Универсальная — тексты, вопросы, анализ',
			tag: 'Текст'
		},
		'kodify-2.0': {
			icon: '💻',
			desc: 'Специализирована на коде',
			tag: 'Код'
		},
		'cotype-preview-32k': {
			icon: '📄',
			desc: 'Длинный контекст до 32k токенов',
			tag: '32k'
		},
		'bge-m3': {
			icon: '🧠',
			desc: 'Эмбеддинги для семантического поиска',
			tag: 'Поиск'
		}
	};

	const getModelCardInfo = (item: { value: string; model?: any }) => {
		return (
			modelInfo[item.value as keyof typeof modelInfo] ?? {
				icon: '🤖',
				desc: item.model?.info?.meta?.description ?? '',
				tag: item.model?.direct
					? 'Direct'
					: item.model?.connection_type === 'external'
						? 'External'
						: ''
			}
		);
	};

	$: itemCardInfo = getModelCardInfo(item);

	const copyLinkHandler = async (model: { id: string }) => {
		const baseUrl = window.location.origin;
		const res = await copyToClipboard(`${baseUrl}/?model=${encodeURIComponent(model.id)}`);

		if (res) {
			toast.success($i18n.t('Copied link to clipboard'));
		} else {
			toast.error($i18n.t('Failed to copy link'));
		}
	};

	let showMenu = false;
</script>

<button
	role="option"
	aria-selected={value === item.value}
	aria-label={$i18n.t('Select {{modelName}} model', { modelName: item.label })}
	class="flex group/item w-full cursor-pointer select-none items-start gap-3 rounded-2xl border px-3 py-3 text-left text-sm text-gray-700 outline-hidden transition-all duration-75 hover:bg-gray-50 dark:text-gray-100 dark:hover:bg-gray-800/70 {value ===
	item.value
		? 'border-blue-600 bg-gray-50 dark:bg-gray-850'
		: index === selectedModelIdx
			? 'border-gray-300 bg-gray-50 dark:border-gray-700 dark:bg-gray-800/70'
			: 'border-transparent'}"
	data-arrow-selected={index === selectedModelIdx}
	data-value={item.value}
	on:click={() => {
		onClick();
	}}
>
	<div class="flex min-w-0 flex-1 items-start gap-3">
		<div
			class="flex size-10 shrink-0 items-center justify-center rounded-2xl bg-gray-100 text-xl dark:bg-gray-800"
		>
			<span aria-hidden="true">{itemCardInfo.icon}</span>
		</div>

		<div class="min-w-0 flex-1">
			<Tooltip content={`${item.label} (${item.value})`} placement="top-start">
				<div class="line-clamp-1 text-sm font-semibold text-gray-900 dark:text-gray-100">
					{item.value}
				</div>
			</Tooltip>

			{#if itemCardInfo.desc}
				<div class="mt-1 line-clamp-2 text-xs leading-5 text-gray-500 dark:text-gray-400">
					{itemCardInfo.desc}
				</div>
			{/if}

			<div class="mt-2 flex flex-wrap items-center gap-2 text-[10px]">
				{#if itemCardInfo.tag}
					<span
						class="rounded-full bg-gray-100 px-2 py-0.5 text-gray-500 dark:bg-gray-800 dark:text-gray-400"
					>
						{itemCardInfo.tag}
					</span>
				{/if}

				{#if item.model.owned_by === 'ollama' && (item.model.ollama?.details?.parameter_size ?? '') !== ''}
					<Tooltip
						content={`${
							item.model.ollama?.details?.quantization_level
								? item.model.ollama?.details?.quantization_level + ' '
								: ''
						}${
							item.model.ollama?.size
								? `(${(item.model.ollama?.size / 1024 ** 3).toFixed(1)}GB)`
								: ''
						}`}
					>
						<span class="text-gray-500 dark:text-gray-400">
							{item.model.ollama?.details?.parameter_size ?? ''}
						</span>
					</Tooltip>
				{/if}

				{#if item.model.ollama?.expires_at && new Date(item.model.ollama?.expires_at * 1000) > new Date()}
					<Tooltip
						content={`${$i18n.t('Unloads {{FROM_NOW}}', {
							FROM_NOW: dayjs(item.model.ollama?.expires_at * 1000).fromNow()
						})}`}
					>
						<span class="inline-flex items-center gap-1 text-gray-500 dark:text-gray-400">
							<span class="relative flex size-2">
								<span
									class="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"
								></span>
								<span class="relative inline-flex rounded-full size-2 bg-green-500"></span>
							</span>
							active
						</span>
					</Tooltip>
				{/if}
			</div>
		</div>
	</div>

	<div class="ml-auto flex shrink-0 items-start gap-1.5 pl-2 pr-1">
		{#if $user?.role === 'admin' && item.model.owned_by === 'ollama' && item.model.ollama?.expires_at && new Date(item.model.ollama?.expires_at * 1000) > new Date()}
			<Tooltip
				content={`${$i18n.t('Eject')}`}
				className="flex-shrink-0 group-hover/item:opacity-100 opacity-0 "
			>
				<button
					class="flex"
					aria-label={$i18n.t('Eject model')}
					on:click={(e) => {
						e.preventDefault();
						e.stopPropagation();
						unloadModelHandler(item.value);
					}}
				>
					<ArrowUpTray className="size-3" />
				</button>
			</Tooltip>
		{/if}

		<ModelItemMenu
			bind:show={showMenu}
			model={item.model}
			{pinModelHandler}
			copyLinkHandler={() => {
				copyLinkHandler(item.model);
			}}
		>
			<button
				aria-label={`${$i18n.t('More Options')}`}
				class="flex rounded-full p-1 text-gray-500 transition hover:bg-gray-100 hover:text-gray-900 dark:hover:bg-gray-800 dark:hover:text-gray-100"
				on:click={(e) => {
					e.preventDefault();
					e.stopPropagation();
					showMenu = !showMenu;
				}}
			>
				<EllipsisHorizontal />
			</button>
		</ModelItemMenu>

		{#if value === item.value}
			<div class="rounded-full bg-blue-600 p-1 text-white">
				<Check className="size-3" />
			</div>
		{/if}
	</div>
</button>
