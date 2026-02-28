"use client";

import { useState, useEffect } from "react";
import StatusBadge from "@/components/StatusBadge";

export default function FilesPage() {
    const [files, setFiles] = useState([]);
    const [isUploading, setIsUploading] = useState(false);
    const [dragActive, setDragActive] = useState(false);

    useEffect(() => {
        fetchFiles();
    }, []);

    const fetchFiles = async () => {
        try {
            const resp = await fetch("http://localhost:8001/api/files");
            const data = await resp.json();
            setFiles(data);
        } catch (err) {
            console.error("Failed to fetch files", err);
        }
    };

    const handleUpload = async (fileList) => {
        setIsUploading(true);
        for (const file of fileList) {
            const formData = new FormData();
            formData.append("file", file);
            try {
                await fetch("http://localhost:8001/api/upload", {
                    method: "POST",
                    body: formData,
                });
            } catch (err) {
                console.error("Upload error", err);
            }
        }
        setIsUploading(false);
        fetchFiles();
    };

    const handleDelete = async (filename) => {
        if (!confirm(`Are you sure you want to delete ${filename}?`)) return;
        try {
            await fetch(`http://localhost:8001/api/files/${filename}`, {
                method: "DELETE",
            });
            fetchFiles();
        } catch (err) {
            console.error("Delete error", err);
        }
    };

    return (
        <div className="flex flex-col h-full bg-background-light dark:bg-background-dark p-6 overflow-y-auto">
            <div className="max-w-[1000px] mx-auto w-full space-y-8">
                {/* Header */}
                <div className="flex flex-col gap-1">
                    <h1 className="text-2xl font-bold dark:text-white">Knowledge Base</h1>
                    <p className="text-slate-500 dark:text-slate-400 text-sm">Upload and manage documents for RAG retrieval.</p>
                </div>

                {/* Upload Area */}
                <div
                    className={`relative border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center transition-all duration-200 ${dragActive
                        ? "border-primary bg-primary/5 scale-[1.01]"
                        : "border-slate-200 dark:border-surface-lighter hover:border-primary/50"
                        }`}
                    onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                    onDragLeave={() => setDragActive(false)}
                    onDrop={(e) => {
                        e.preventDefault();
                        setDragActive(false);
                        if (e.dataTransfer.files) handleUpload(e.dataTransfer.files);
                    }}
                >
                    <div className="size-12 rounded-full bg-primary/10 text-primary flex items-center justify-center mb-4">
                        <span className="material-symbols-outlined text-2xl">upload_file</span>
                    </div>
                    <div className="text-center">
                        <p className="text-sm font-semibold dark:text-white mb-1">
                            {isUploading ? "Uploading..." : "Click to upload or drag & drop"}
                        </p>
                        <p className="text-xs text-slate-500 dark:text-slate-400">PDF, TXT, DOCX, or MD (Max 10MB)</p>
                    </div>
                    <input
                        type="file"
                        multiple
                        className="absolute inset-0 opacity-0 cursor-pointer"
                        onChange={(e) => handleUpload(e.target.files)}
                        disabled={isUploading}
                    />
                </div>

                {/* File Table */}
                <div className="bg-white dark:bg-surface-dark rounded-xl shadow-sm border border-slate-200 dark:border-surface-lighter overflow-hidden">
                    <table className="w-full text-left border-collapse">
                        <thead>
                            <tr className="border-b border-slate-100 dark:border-surface-lighter bg-slate-50/50 dark:bg-surface-lighter/30">
                                <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">File Name</th>
                                <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 text-center">Status</th>
                                <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 text-right">Action</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 dark:divide-surface-lighter">
                            {files.length === 0 ? (
                                <tr>
                                    <td colSpan="3" className="px-6 py-12 text-center text-slate-400 dark:text-slate-600 text-sm">
                                        No documents uploaded yet.
                                    </td>
                                </tr>
                            ) : (
                                files.map((file, idx) => (
                                    <tr key={idx} className="hover:bg-slate-50/50 dark:hover:bg-surface-lighter/20 transition-colors group">
                                        <td className="px-6 py-4">
                                            <div className="flex items-center gap-3">
                                                <span className="material-symbols-outlined text-slate-400 group-hover:text-primary transition-colors">description</span>
                                                <span className="text-sm font-medium dark:text-slate-200">{file.filename}</span>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4">
                                            <div className="flex justify-center">
                                                <StatusBadge status={file.status?.toLowerCase() || "ready"} />
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 text-right">
                                            <button
                                                onClick={() => handleDelete(file.filename)}
                                                className="p-2 text-slate-400 hover:text-rose-500 hover:bg-rose-50 dark:hover:bg-rose-500/10 rounded-lg transition-all"
                                            >
                                                <span className="material-symbols-outlined text-xl">delete</span>
                                            </button>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
