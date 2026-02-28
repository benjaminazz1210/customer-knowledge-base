"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Header() {
    const pathname = usePathname();

    const navItems = [
        { name: "Chat", path: "/" },
        { name: "Files", path: "/files" },
    ];

    return (
        <header className="w-full bg-background-light dark:bg-background-dark border-b border-slate-200 dark:border-surface-lighter flex-none z-10 transition-colors">
            <div className="max-w-[1200px] mx-auto w-full px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                    <div className="flex items-center gap-4">
                        <div className="size-8 flex items-center justify-center rounded-lg bg-primary/10 text-primary">
                            <span className="material-symbols-outlined text-2xl">dataset</span>
                        </div>
                        <div className="font-bold text-xl tracking-tight text-slate-900 dark:text-white uppercase">NexusAI</div>
                    </div>

                    <nav className="flex h-full gap-2 md:gap-8">
                        {navItems.map((item) => (
                            <Link
                                key={item.path}
                                href={item.path}
                                className={`flex flex-col items-center justify-center border-b-2 h-full px-4 transition-all duration-200 ${pathname === item.path
                                        ? "border-primary text-slate-900 dark:text-white"
                                        : "border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
                                    }`}
                            >
                                <span className="text-sm font-semibold leading-normal tracking-wide">
                                    {item.name}
                                </span>
                            </Link>
                        ))}
                    </nav>

                    <div className="flex items-center gap-3">
                        <button className="p-2 text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-white rounded-full transition-colors">
                            <span className="material-symbols-outlined text-xl">settings</span>
                        </button>
                        <div
                            className="bg-primary/20 rounded-full w-8 h-8 shrink-0 flex items-center justify-center text-primary ring-2 ring-primary/10 cursor-pointer"
                            title="User Profile"
                        >
                            <span className="material-symbols-outlined text-xl">person</span>
                        </div>
                    </div>
                </div>
            </div>
        </header>
    );
}
