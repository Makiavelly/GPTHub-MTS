<script lang="ts">
	import { tick, onMount } from 'svelte';

	import { mobile } from '$lib/stores';

	type MessageContentPart = {
		type?: string;
		text?: string;
	};

	type QuickNavMessage = {
		id: string;
		role?: string;
		content?: string | MessageContentPart[];
		files?: { name?: string }[];
	};

	export let messages: QuickNavMessage[] = [];
	export let containerId = 'messages-container';

	let rootElement: HTMLDivElement;
	let containerElement: HTMLElement | null = null;
	let expandedScrollElement: HTMLDivElement;
	let activeMessageId: string | null = null;
	let isHovered = false;
	let isMobileExpanded = false;
	let frameId: number | null = null;
	let userMessages: QuickNavMessage[] = [];
	let collapsedMessages: QuickNavMessage[] = [];
	let isExpanded = false;

	const COLLAPSED_VISIBLE_COUNT = 9;

	$: userMessages = messages.filter((message) => message?.role === 'user');
	$: isExpanded = $mobile ? isMobileExpanded : isHovered;
	$: {
		if (userMessages.length <= COLLAPSED_VISIBLE_COUNT) {
			collapsedMessages = userMessages;
		} else {
			const resolvedActiveIndex = userMessages.findIndex((message) => message.id === activeMessageId);
			const activeIndex = resolvedActiveIndex >= 0 ? resolvedActiveIndex : 0;
			let startIndex =
				Math.floor(activeIndex / COLLAPSED_VISIBLE_COUNT) * COLLAPSED_VISIBLE_COUNT;
			let endIndex = Math.min(startIndex + COLLAPSED_VISIBLE_COUNT, userMessages.length);

			if (endIndex - startIndex < COLLAPSED_VISIBLE_COUNT) {
				startIndex = Math.max(0, endIndex - COLLAPSED_VISIBLE_COUNT);
				endIndex = userMessages.length;
			}

			collapsedMessages = userMessages.slice(startIndex, endIndex);
		}
	}

	const getPreview = (message: QuickNavMessage) => {
		const content =
			typeof message?.content === 'string'
				? message.content
				: Array.isArray(message?.content)
					? message.content
							.filter((part: MessageContentPart) => part?.type === 'text')
							.map((part: MessageContentPart) => part?.text ?? '')
							.join(' ')
					: '';

		const normalized = content.replace(/\s+/g, ' ').trim();
		if (normalized.length > 52) {
			return `${normalized.slice(0, 52)}...`;
		}

		if (normalized) {
			return normalized;
		}

		if ((message?.files ?? []).length > 0) {
			return message.files?.[0]?.name ?? 'Файл';
		}

		return 'Сообщение';
	};

	const updateActiveMessage = () => {
		if (!containerElement || userMessages.length <= 1) {
			activeMessageId = null;
			return;
		}

		const containerRect = containerElement.getBoundingClientRect();
		let nextActiveMessageId = userMessages[0]?.id ?? null;
		let minDistance = Number.POSITIVE_INFINITY;

		for (const message of userMessages) {
			const element = document.getElementById(`message-${message.id}`);
			if (!element) {
				continue;
			}

			const rect = element.getBoundingClientRect();
			const distance = Math.abs(rect.top - containerRect.top);

			if (distance < minDistance) {
				minDistance = distance;
				nextActiveMessageId = message.id;
			}
		}

		activeMessageId = nextActiveMessageId;
	};

	const scheduleActiveMessageUpdate = () => {
		if (frameId !== null) {
			cancelAnimationFrame(frameId);
		}

		frameId = requestAnimationFrame(() => {
			frameId = null;
			updateActiveMessage();
		});
	};

	const scrollToMessage = async (messageId: string) => {
		const element = document.getElementById(`message-${messageId}`);
		if (!element) {
			return;
		}

		element.scrollIntoView({
			behavior: 'smooth',
			block: 'start'
		});

		activeMessageId = messageId;

		if ($mobile) {
			isMobileExpanded = false;
		}
	};

	const handleDocumentClick = (event: MouseEvent) => {
		if (!$mobile || !isMobileExpanded || !rootElement) {
			return;
		}

		if (!rootElement.contains(event.target as Node)) {
			isMobileExpanded = false;
		}
	};

	const syncExpandedScroll = () => {
		if (!isExpanded || !expandedScrollElement || !activeMessageId) {
			return;
		}

		const activeButton = Array.from(
			expandedScrollElement.querySelectorAll<HTMLButtonElement>('[data-message-id]')
		).find((element) => element.dataset.messageId === activeMessageId);

		activeButton?.scrollIntoView({
			block: 'nearest'
		});
	};

	onMount(() => {
		containerElement = document.getElementById(containerId);

		if (containerElement) {
			containerElement.addEventListener('scroll', scheduleActiveMessageUpdate, { passive: true });
		}

		window.addEventListener('resize', scheduleActiveMessageUpdate);
		document.addEventListener('click', handleDocumentClick);
		scheduleActiveMessageUpdate();

		return () => {
			if (containerElement) {
				containerElement.removeEventListener('scroll', scheduleActiveMessageUpdate);
			}

			window.removeEventListener('resize', scheduleActiveMessageUpdate);
			document.removeEventListener('click', handleDocumentClick);

			if (frameId !== null) {
				cancelAnimationFrame(frameId);
			}
		};
	});

	$: if (typeof window !== 'undefined' && userMessages.length > 1) {
		tick().then(() => {
			scheduleActiveMessageUpdate();
		});
	}

	$: if (typeof window !== 'undefined' && isExpanded && activeMessageId) {
		tick().then(() => {
			syncExpandedScroll();
		});
	}
</script>

{#if userMessages.length > 1}
	<div
		bind:this={rootElement}
		role="navigation"
		aria-label="User message navigation"
		class="fixed right-2 top-1/2 z-20 -translate-y-1/2 md:right-5"
		on:mouseenter={() => {
			if (!$mobile) {
				isHovered = true;
			}
		}}
		on:mouseleave={() => {
			if (!$mobile) {
				isHovered = false;
			}
		}}
	>
		{#if isExpanded}
			<div
				class="w-48 max-w-[70vw] rounded-3xl border border-gray-200/80 bg-white/92 p-2.5 shadow-2xl backdrop-blur-md dark:border-gray-700/70 dark:bg-gray-900/92"
			>
				<div
					bind:this={expandedScrollElement}
					class="quick-nav-scroll flex flex-col gap-1 overflow-y-auto overscroll-contain pr-1"
					style="max-height: min(18rem, 55vh);"
				>
					{#each userMessages as message}
						<button
							type="button"
							data-message-id={message.id}
							class="flex w-full items-center gap-2 rounded-2xl px-3 py-2 text-left text-xs transition hover:bg-blue-50 dark:hover:bg-gray-800"
							on:click={() => scrollToMessage(message.id)}
						>
							<span
								class="min-w-0 flex-1 truncate {activeMessageId === message.id
									? 'text-blue-500 dark:text-blue-400'
									: 'text-gray-500 dark:text-gray-400'}"
							>
								{getPreview(message)}
							</span>
							<span
								class="h-0.5 w-3 shrink-0 rounded-full {activeMessageId === message.id
									? 'bg-blue-500'
									: 'bg-gray-300 dark:bg-gray-600'}"
							></span>
						</button>
					{/each}
				</div>
			</div>
		{:else}
			<button
				type="button"
				class="flex cursor-pointer flex-col items-end gap-4 rounded-full px-1 py-2"
				aria-label="Open message navigation"
				on:click={() => {
					if ($mobile) {
						isMobileExpanded = true;
					}
				}}
			>
				{#each collapsedMessages as message}
					<span
						class="h-0.5 w-3 rounded-full transition {activeMessageId === message.id
							? 'bg-blue-500'
							: 'bg-gray-300 dark:bg-gray-500'}"
					></span>
				{/each}
			</button>
		{/if}
	</div>
{/if}

<style>
	.quick-nav-scroll {
		scrollbar-width: thin;
		scrollbar-color: rgba(96, 165, 250, 0.9) rgba(148, 163, 184, 0.12);
	}

	.quick-nav-scroll::-webkit-scrollbar {
		width: 6px;
	}

	.quick-nav-scroll::-webkit-scrollbar-track {
		background: rgba(148, 163, 184, 0.12);
		border-radius: 999px;
	}

	.quick-nav-scroll::-webkit-scrollbar-thumb {
		background: rgba(96, 165, 250, 0.9);
		border-radius: 999px;
	}

	.quick-nav-scroll::-webkit-scrollbar-thumb:hover {
		background: rgba(59, 130, 246, 1);
	}
</style>
