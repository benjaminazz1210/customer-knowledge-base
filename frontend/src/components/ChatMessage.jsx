import { useState } from 'react';

export default function ChatMessage({ message, isAi, sources, isThinking, isStreaming }) {
    const [expanded, setExpanded] = useState(false);
    const [copied, setCopied] = useState(false);
    const showThinking = isAi && isThinking && !message;

    const handleCopy = async () => {
        if (!message) return;

        try {
            if (navigator?.clipboard?.writeText) {
                await navigator.clipboard.writeText(message);
            } else {
                const textArea = document.createElement('textarea');
                textArea.value = message;
                textArea.style.position = 'fixed';
                textArea.style.left = '-9999px';
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();
                document.execCommand('copy');
                document.body.removeChild(textArea);
            }
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
        } catch (err) {
            console.error('复制失败', err);
        }
    };

    return (
        <div className={`flex items-start gap-4 mb-6 group animate-fade-in ${isAi ? "" : "flex-row-reverse"}`}>
            <div className={`rounded-full w-10 h-10 shrink-0 flex items-center justify-center text-xl ${isAi ? "bg-primary/10" : "bg-primary"
                }`}>
                {isAi ? "🤖" : "🐷"}
            </div>
            <div className={`flex flex-col gap-2 max-w-[85%] ${isAi ? "" : "items-end"}`}>
                <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                        {isAi ? "猪你好运" : "你"}
                    </span>
                </div>
                <div className={`rounded-2xl px-5 py-4 shadow-sm border leading-relaxed text-[15px] ${isAi
                    ? "bg-white dark:bg-surface-lighter rounded-tl-none border-slate-100 dark:border-transparent text-slate-800 dark:text-slate-200"
                    : "bg-primary text-white rounded-tr-none border-transparent"
                    }`}>
                    {showThinking ? (
                        <div className="flex items-center gap-2 min-h-6">
                            <span className="text-slate-500 dark:text-slate-300 text-sm font-medium">思考中</span>
                            <div className="flex gap-1.5 items-center">
                                <div className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-pulse" />
                                <div className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-pulse delay-75" />
                                <div className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-pulse delay-150" />
                            </div>
                        </div>
                    ) : (
                        <p className="whitespace-pre-wrap">
                            {message}
                            {isAi && isStreaming && message ? (
                                <span className="inline-block align-middle ml-1 h-4 w-[2px] bg-primary/70 animate-pulse rounded-full" />
                            ) : null}
                        </p>
                    )}
                </div>
                {message && (
                    <div className={`w-full flex ${isAi ? "justify-start" : "justify-end"}`}>
                        <button
                            type="button"
                            onClick={handleCopy}
                            className="inline-flex items-center gap-1.5 text-xs text-slate-500 hover:text-primary transition-colors"
                        >
                            <span className="material-symbols-outlined text-[14px]">
                                {copied ? "check" : "content_copy"}
                            </span>
                            {copied ? "已复制" : "复制"}
                        </button>
                    </div>
                )}

                {isAi && sources && sources.length > 0 && (
                    <div className="mt-2 w-full">
                        <button
                            onClick={() => setExpanded(!expanded)}
                            className="flex items-center gap-1.5 text-xs text-primary hover:text-primary/80 font-medium transition-colors"
                        >
                            <span className="material-symbols-outlined text-[16px]">
                                {expanded ? "expand_less" : "expand_more"}
                            </span>
                            {expanded ? "收起参考来源" : `查看 ${sources.length} 个参考片段`}
                        </button>

                        {expanded && (
                            <div className="flex flex-col gap-2 mt-3">
                                {sources.map((source, idx) => {
                                    // Handle legacy array of strings versus new array of objects
                                    const isObject = typeof source === 'object' && source !== null;
                                    const sourceFile = isObject ? source.source_file : source;
                                    const score = isObject ? source.score : null;
                                    const content = isObject ? source.content : "遗留格式：无内容详情";

                                    return (
                                        <div key={idx} className="bg-slate-50 dark:bg-surface-lighter/30 rounded-lg p-3 border border-slate-100 dark:border-surface-lighter animate-fade-in">
                                            <div className="flex justify-between items-center mb-1.5">
                                                <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-700 dark:text-slate-300">
                                                    <span className="material-symbols-outlined text-sm text-primary">description</span>
                                                    {sourceFile}
                                                </div>
                                                {score !== null && (
                                                    <span className="bg-primary/10 text-primary text-[10px] px-1.5 py-0.5 rounded font-mono font-medium">
                                                        相似度: {(score * 100).toFixed(1)}%
                                                    </span>
                                                )}
                                            </div>
                                            <p className="text-xs text-slate-600 dark:text-slate-400 line-clamp-3 leading-relaxed">
                                                {content}
                                            </p>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
