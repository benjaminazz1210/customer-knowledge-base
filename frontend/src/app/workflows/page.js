"use client";

import { useEffect, useMemo, useState } from "react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8001";

export default function WorkflowsPage() {
  const [prompt, setPrompt] = useState("请基于现有知识库，生成一个1000字的实施方案文档。");
  const [fileType, setFileType] = useState("auto");
  const [targetWords, setTargetWords] = useState(1000);
  const [targetSlides, setTargetSlides] = useState(7);
  const [useRag, setUseRag] = useState(true);

  const [isGenerating, setIsGenerating] = useState(false);
  const [isRevising, setIsRevising] = useState(false);
  const [feedback, setFeedback] = useState("第三页的架构图不够详细，加上门禁系统的说明。");
  const [result, setResult] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [error, setError] = useState("");

  const effectiveDownloadUrl = useMemo(() => {
    if (!result?.download_url) return "";
    return `${API_BASE_URL}${result.download_url}`;
  }, [result]);

  const fetchJobs = async () => {
    try {
      const resp = await fetch(`${API_BASE_URL}/api/workflows/jobs?limit=20`);
      if (!resp.ok) return;
      const data = await resp.json();
      if (Array.isArray(data)) setJobs(data);
    } catch (err) {
      console.error("获取工作流任务失败", err);
    }
  };

  useEffect(() => {
    fetchJobs();
  }, []);

  const handleGenerate = async (e) => {
    e.preventDefault();
    if (!prompt.trim() || isGenerating) return;
    setError("");
    setIsGenerating(true);
    try {
      const payload = {
        prompt: prompt.trim(),
        file_type: fileType === "auto" ? null : fileType,
        target_words: fileType === "pptx" ? null : Number(targetWords) || null,
        target_slides: fileType === "docx" ? null : Number(targetSlides) || null,
        use_rag: useRag,
      };
      const resp = await fetch(`${API_BASE_URL}/api/workflows/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data?.detail || `HTTP ${resp.status}`);
      }
      setResult(data);
      await fetchJobs();
    } catch (err) {
      console.error("生成失败", err);
      setError(`生成失败：${String(err.message || err)}`);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleRevise = async (e) => {
    e.preventDefault();
    if (!result?.job_id || !feedback.trim() || isRevising) return;
    setError("");
    setIsRevising(true);
    try {
      const resp = await fetch(`${API_BASE_URL}/api/workflows/revise`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_id: result.job_id,
          feedback: feedback.trim(),
        }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data?.detail || `HTTP ${resp.status}`);
      }
      setResult(data);
      await fetchJobs();
    } catch (err) {
      console.error("改稿失败", err);
      setError(`改稿失败：${String(err.message || err)}`);
    } finally {
      setIsRevising(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-background-light dark:bg-background-dark p-6 overflow-y-auto">
      <div className="max-w-[1100px] mx-auto w-full space-y-8">
        <div className="space-y-1">
          <h1 className="text-2xl font-bold dark:text-white">Multi-Agent 工作流</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Phase 3/4: 需求分析 → 检索增强 → 文档生成 → 人机协同改稿（HITL）。
          </p>
        </div>

        <form
          onSubmit={handleGenerate}
          className="bg-white dark:bg-surface-dark rounded-xl shadow-sm border border-slate-200 dark:border-surface-lighter p-5 space-y-4"
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="md:col-span-2">
              <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">生成需求</label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={4}
                className="w-full rounded-lg border border-slate-200 dark:border-surface-lighter bg-white dark:bg-surface-lighter/30 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
                placeholder="例如：生成7页PPT，主题是智慧园区安防改造方案。"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">文件类型</label>
              <select
                value={fileType}
                onChange={(e) => setFileType(e.target.value)}
                className="w-full rounded-lg border border-slate-200 dark:border-surface-lighter bg-white dark:bg-surface-lighter/30 px-3 py-2 text-sm"
              >
                <option value="auto">自动识别</option>
                <option value="docx">Word (.docx)</option>
                <option value="pptx">PPT (.pptx)</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">目标字数（Word）</label>
              <input
                type="number"
                min={100}
                max={50000}
                value={targetWords}
                onChange={(e) => setTargetWords(e.target.value)}
                className="w-full rounded-lg border border-slate-200 dark:border-surface-lighter bg-white dark:bg-surface-lighter/30 px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">目标页数（PPT）</label>
              <input
                type="number"
                min={1}
                max={60}
                value={targetSlides}
                onChange={(e) => setTargetSlides(e.target.value)}
                className="w-full rounded-lg border border-slate-200 dark:border-surface-lighter bg-white dark:bg-surface-lighter/30 px-3 py-2 text-sm"
              />
            </div>
            <div className="flex items-center gap-2 mt-4">
              <input
                id="useRag"
                type="checkbox"
                checked={useRag}
                onChange={(e) => setUseRag(e.target.checked)}
              />
              <label htmlFor="useRag" className="text-sm text-slate-600 dark:text-slate-300">
                启用知识库检索（Hybrid Search）
              </label>
            </div>
          </div>
          <button
            type="submit"
            disabled={isGenerating}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-white text-sm font-medium hover:bg-primary/90 disabled:opacity-60"
          >
            <span className="material-symbols-outlined text-[18px]">{isGenerating ? "hourglass_empty" : "auto_awesome"}</span>
            {isGenerating ? "生成中..." : "生成文件"}
          </button>
        </form>

        {error && (
          <div className="rounded-lg border border-rose-200 bg-rose-50 text-rose-700 px-4 py-3 text-sm">
            {error}
          </div>
        )}

        {result && (
          <div className="bg-white dark:bg-surface-dark rounded-xl shadow-sm border border-slate-200 dark:border-surface-lighter p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="text-sm text-slate-500 dark:text-slate-400">任务 ID: {result.job_id}</div>
                <div className="text-sm text-slate-500 dark:text-slate-400">版本: v{result.version}</div>
              </div>
              {effectiveDownloadUrl && (
                <a
                  href={effectiveDownloadUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 text-white text-sm font-medium hover:bg-emerald-700"
                >
                  <span className="material-symbols-outlined text-[18px]">download</span>
                  下载最新文件
                </a>
              )}
            </div>

            <form onSubmit={handleRevise} className="space-y-3">
              <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400">
                人机协同改稿（HITL）
              </label>
              <input
                value={feedback}
                onChange={(e) => setFeedback(e.target.value)}
                className="w-full rounded-lg border border-slate-200 dark:border-surface-lighter bg-white dark:bg-surface-lighter/30 px-3 py-2 text-sm"
                placeholder="例如：第三页的架构图不够详细，加上门禁系统的说明。"
              />
              <button
                type="submit"
                disabled={isRevising}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-900 text-white text-sm font-medium hover:bg-slate-800 disabled:opacity-60"
              >
                <span className="material-symbols-outlined text-[18px]">{isRevising ? "hourglass_empty" : "edit_square"}</span>
                {isRevising ? "改稿中..." : "按反馈改稿并生成新版本"}
              </button>
            </form>
          </div>
        )}

        <div className="bg-white dark:bg-surface-dark rounded-xl shadow-sm border border-slate-200 dark:border-surface-lighter p-5">
          <h2 className="text-sm font-semibold mb-3 text-slate-700 dark:text-slate-300">最近任务</h2>
          <div className="space-y-2">
            {jobs.length === 0 && <p className="text-sm text-slate-400">暂无任务。</p>}
            {jobs.map((job) => (
              <div
                key={job.job_id}
                className="flex flex-wrap items-center justify-between gap-3 px-3 py-2 rounded-lg border border-slate-100 dark:border-surface-lighter"
              >
                <div className="min-w-0">
                  <p className="text-xs text-slate-500 dark:text-slate-400">{job.job_id} · v{job.current_version}</p>
                  <p className="text-sm text-slate-700 dark:text-slate-300 truncate max-w-[680px]">{job.prompt}</p>
                </div>
                {job.latest_download_url && (
                  <a
                    href={`${API_BASE_URL}${job.latest_download_url}`}
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs text-primary hover:underline"
                  >
                    下载
                  </a>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
