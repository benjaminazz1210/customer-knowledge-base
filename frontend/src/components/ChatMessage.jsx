export default function ChatMessage({ message, isAi, sources }) {
    return (
        <div className={`flex items-start gap-4 mb-6 group animate-fade-in ${isAi ? "" : "flex-row-reverse"}`}>
            <div className={`rounded-full w-10 h-10 shrink-0 flex items-center justify-center ${isAi ? "bg-primary/10 text-primary" : "bg-primary text-white"
                }`}>
                <span className="material-symbols-outlined text-xl">
                    {isAi ? "smart_toy" : "person"}
                </span>
            </div>
            <div className={`flex flex-col gap-2 max-w-[85%] ${isAi ? "" : "items-end"}`}>
                <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                        {isAi ? "NexusAI Assistant" : "You"}
                    </span>
                </div>
                <div className={`rounded-2xl px-5 py-4 shadow-sm border leading-relaxed text-[15px] ${isAi
                        ? "bg-white dark:bg-surface-lighter rounded-tl-none border-slate-100 dark:border-transparent text-slate-800 dark:text-slate-200"
                        : "bg-primary text-white rounded-tr-none border-transparent"
                    }`}>
                    <p className="whitespace-pre-wrap">{message}</p>
                </div>

                {isAi && sources && sources.length > 0 && (
                    <div className="flex flex-wrap items-center gap-2 mt-1">
                        {sources.map((source, idx) => (
                            <div
                                key={idx}
                                className="flex items-center gap-2 pl-2 pr-3 py-1.5 bg-surface-dark dark:bg-surface-lighter/50 border border-slate-200 dark:border-transparent rounded-lg text-xs font-medium text-slate-600 dark:text-slate-300"
                            >
                                <span className="material-symbols-outlined text-primary text-sm">description</span>
                                <span>{source}</span>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
