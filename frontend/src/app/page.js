"use client";

import { useState, useRef, useEffect } from "react";
import ChatMessage from "@/components/ChatMessage";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "";
const ARCHIVE_STORAGE_KEY = "nexusai_chat_archives_v1";

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showNewChatConfirm, setShowNewChatConfirm] = useState(false);
  const [showHistoryPanel, setShowHistoryPanel] = useState(false);
  const [archivedChats, setArchivedChats] = useState([]);
  const scrollRef = useRef(null);
  const tokenQueueRef = useRef("");
  const streamEndedRef = useRef(false);
  const flusherRunningRef = useRef(false);
  const renderDoneResolverRef = useRef(null);

  const defaultMessage = { text: "你好！我是猪你好运，你的智能助手。你可以将文档上传到知识库，然后向我提问。", isAi: true, sources: [] };

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const resp = await fetch(`${API_BASE_URL}/api/history`);
      if (resp.ok) {
        const data = await resp.json();
        if (data && data.length > 0) {
          setMessages(data);
          return;
        }
      }
    } catch (err) {
      console.error("获取历史记录失败", err);
    }
    setMessages([defaultMessage]);
  };

  useEffect(() => {
    try {
      const raw = localStorage.getItem(ARCHIVE_STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        setArchivedChats(parsed);
      }
    } catch (err) {
      console.error("加载聊天归档失败", err);
    }
  }, []);

  const persistArchivedChats = (nextChats) => {
    const normalized = [...nextChats].slice(0, 30);
    setArchivedChats(normalized);
    try {
      localStorage.setItem(ARCHIVE_STORAGE_KEY, JSON.stringify(normalized));
    } catch (err) {
      console.error("保存聊天归档失败", err);
    }
  };

  const buildArchiveTitle = (records) => {
    const firstUser = records.find((item) => !item?.isAi && item?.text?.trim());
    if (firstUser) return firstUser.text.trim().slice(0, 32);
    return `对话 ${new Date().toLocaleString()}`;
  };

  const archiveCurrentConversation = () => {
    const hasConversation = messages.length > 1;
    if (!hasConversation) return;

    const archiveItem = {
      id: `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      title: buildArchiveTitle(messages),
      savedAt: Date.now(),
      messages
    };
    persistArchivedChats([archiveItem, ...archivedChats]);
  };

  useEffect(() => {
    // Only save if there's actual conversation beyond the default greeting
    if (messages.length > 1) {
      fetch(`${API_BASE_URL}/api/history`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages })
      }).catch(err => console.error("保存历史记录失败", err));
    }
  }, [messages]);

  const handleNewChat = async ({ archive = false } = {}) => {
    try {
      if (archive) {
        archiveCurrentConversation();
      }
      await fetch(`${API_BASE_URL}/api/history`, { method: "DELETE" });
      setMessages([defaultMessage]);
      setShowNewChatConfirm(false);
      setShowHistoryPanel(false);
    } catch (err) {
      console.error("清空历史记录失败", err);
    }
  };

  const handleRestoreArchive = async (archiveId) => {
    const item = archivedChats.find((chat) => chat.id === archiveId);
    if (!item) return;

    setMessages(item.messages || [defaultMessage]);
    setShowHistoryPanel(false);

    try {
      await fetch(`${API_BASE_URL}/api/history`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: item.messages || [defaultMessage] })
      });
    } catch (err) {
      console.error("恢复聊天记录失败", err);
    }
  };

  const handleDeleteArchive = (archiveId) => {
    persistArchivedChats(archivedChats.filter((chat) => chat.id !== archiveId));
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const updateLastAiMessage = (updater) => {
    setMessages((prev) => {
      if (prev.length === 0) return prev;
      const newMsgs = [...prev];
      const lastIdx = newMsgs.length - 1;
      const last = { ...newMsgs[lastIdx] };
      updater(last);
      newMsgs[lastIdx] = last;
      return newMsgs;
    });
  };

  const ensureTokenFlusher = () => {
    if (flusherRunningRef.current) return;
    flusherRunningRef.current = true;

    (async () => {
      while (!streamEndedRef.current || tokenQueueRef.current.length > 0) {
        const pending = tokenQueueRef.current.length;
        if (pending <= 0) {
          await sleep(16);
          continue;
        }

        const batchSize = pending > 600 ? 8 : pending > 300 ? 5 : pending > 120 ? 3 : 2;
        const chunk = tokenQueueRef.current.slice(0, batchSize);
        tokenQueueRef.current = tokenQueueRef.current.slice(batchSize);

        updateLastAiMessage((last) => {
          last.text = `${last.text || ""}${chunk}`;
          last.isThinking = false;
          last.isStreaming = true;
        });

        const pause = /[。！？.!?；;:：\n]$/.test(chunk) ? 90 : 35;
        await sleep(pause);
      }

      updateLastAiMessage((last) => {
        last.isStreaming = false;
        if (!last.text) {
          last.isThinking = false;
        }
      });

      flusherRunningRef.current = false;
      if (renderDoneResolverRef.current) {
        renderDoneResolverRef.current();
        renderDoneResolverRef.current = null;
      }
    })();
  };

  const waitRenderComplete = async () => {
    if (!flusherRunningRef.current && tokenQueueRef.current.length === 0) return;
    await new Promise((resolve) => {
      renderDoneResolverRef.current = resolve;
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const userInput = input.trim();
    if (!userInput || isLoading) return;

    const userMsg = { text: userInput, isAi: false };
    const aiMsg = { text: "", isAi: true, sources: [], isThinking: true, isStreaming: true };
    setMessages(prev => [...prev, userMsg, aiMsg]);
    setInput("");
    setIsLoading(true);
    tokenQueueRef.current = "";
    streamEndedRef.current = false;
    renderDoneResolverRef.current = null;
    ensureTokenFlusher();

    const controller = new AbortController();

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream"
        },
        body: JSON.stringify({ message: userInput }),
        signal: controller.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      if (!response.body) {
        throw new Error("响应中缺少可读流");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let streamDone = false;

      while (!streamDone) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        // Keep the last (potentially incomplete) line in the buffer
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const dataStr = line.slice(6).trim();
          if (dataStr === "[DONE]") {
            streamDone = true;
            reader.cancel();
            break;
          }
          try {
            const data = JSON.parse(dataStr);
            if (data.token) {
              tokenQueueRef.current += data.token;
            } else if (data.sources) {
              updateLastAiMessage((last) => {
                last.sources = data.sources;
              });
            }
          } catch (err) {
            console.error("Error parsing SSE data", err);
          }
        }
      }

      // Flush decoder tail to avoid dropping a partial final chunk.
      buffer += decoder.decode();
      if (buffer.trim().startsWith("data: ")) {
        const dataStr = buffer.trim().slice(6).trim();
        if (dataStr !== "[DONE]") {
          try {
            const data = JSON.parse(dataStr);
            if (data.token) {
              tokenQueueRef.current += data.token;
            } else if (data.sources) {
              updateLastAiMessage((last) => {
                last.sources = data.sources;
              });
            }
          } catch (err) {
            console.error("Error parsing SSE tail data", err);
          }
        }
      }
      streamEndedRef.current = true;
      await waitRenderComplete();
    } catch (error) {
      console.error("Chat error:", error);
      streamEndedRef.current = true;
      tokenQueueRef.current = "";
      setMessages((prev) => {
        const newMsgs = [...prev];
        const last = { ...newMsgs[newMsgs.length - 1] };
        last.text = "抱歉，发生了错误。请检查后端服务是否正常运行。";
        last.isThinking = false;
        last.isStreaming = false;
        newMsgs[newMsgs.length - 1] = last;
        return newMsgs;
      });
    } finally {
      controller.abort();
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full relative">
      {/* 新对话按钮区域 */}
      <div className="absolute top-4 right-4 z-30 flex flex-col items-end gap-2">
        {showHistoryPanel && (
          <div className="w-[360px] max-w-[92vw] bg-white dark:bg-surface-lighter rounded-xl shadow-xl border border-slate-200 dark:border-surface-lighter p-3 animate-fade-in">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold text-slate-700 dark:text-slate-200">聊天记录</span>
              <button
                onClick={() => setShowHistoryPanel(false)}
                className="text-xs text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors"
              >
                关闭
              </button>
            </div>
            {archivedChats.length === 0 ? (
              <p className="text-xs text-slate-500 dark:text-slate-400 py-4 text-center">暂无已保存的聊天记录</p>
            ) : (
              <div className="max-h-[320px] overflow-y-auto space-y-2 pr-1">
                {archivedChats.map((chat) => (
                  <div key={chat.id} className="rounded-lg border border-slate-200 dark:border-surface-lighter/60 p-2 bg-slate-50 dark:bg-surface-lighter/30">
                    <p className="text-xs font-medium text-slate-700 dark:text-slate-200 truncate">{chat.title}</p>
                    <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
                      {new Date(chat.savedAt).toLocaleString()} · {chat.messages?.length || 0} 条
                    </p>
                    <div className="mt-2 flex items-center gap-2">
                      <button
                        onClick={() => handleRestoreArchive(chat.id)}
                        className="text-[11px] px-2 py-1 rounded bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
                      >
                        恢复
                      </button>
                      <button
                        onClick={() => handleDeleteArchive(chat.id)}
                        className="text-[11px] px-2 py-1 rounded bg-rose-50 text-rose-500 hover:bg-rose-100 transition-colors"
                      >
                        删除
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {showNewChatConfirm && (
          <div className="flex items-center gap-2 bg-white dark:bg-surface-lighter px-3 py-1.5 rounded-full shadow-lg border border-slate-200 dark:border-surface-lighter animate-fade-in">
            <span className="text-xs text-slate-500 font-medium">确认清空？</span>
            <button onClick={() => setShowNewChatConfirm(false)} className="text-xs text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors">取消</button>
            <button onClick={() => handleNewChat({ archive: true })} className="text-xs text-primary hover:text-primary/80 font-medium transition-colors">保存并新建</button>
            <button onClick={() => handleNewChat({ archive: false })} className="text-xs text-rose-500 hover:text-rose-600 font-medium transition-colors">清空并新建</button>
          </div>
        )}

        <div className="flex items-center gap-2">
          {!showNewChatConfirm && (
            <button
              onClick={() => {
                setShowHistoryPanel(false);
                setShowNewChatConfirm(true);
              }}
              className="flex items-center justify-center gap-1.5 bg-white/80 dark:bg-surface-lighter/80 backdrop-blur-sm px-4 py-2 rounded-full text-sm font-medium text-slate-600 dark:text-slate-300 shadow-sm border border-slate-200/60 dark:border-surface-lighter/60 hover:text-primary hover:border-primary/30 transition-all hover:shadow-md"
            >
              <span className="material-symbols-outlined text-[18px]">add_comment</span>
              <span className="hidden sm:inline">新对话</span>
            </button>
          )}
          <button
            onClick={() => {
              setShowNewChatConfirm(false);
              setShowHistoryPanel((prev) => !prev);
            }}
            className="flex items-center justify-center gap-1.5 bg-white/80 dark:bg-surface-lighter/80 backdrop-blur-sm px-4 py-2 rounded-full text-sm font-medium text-slate-600 dark:text-slate-300 shadow-sm border border-slate-200/60 dark:border-surface-lighter/60 hover:text-primary hover:border-primary/30 transition-all hover:shadow-md"
          >
            <span className="material-symbols-outlined text-[18px]">history</span>
            <span className="hidden sm:inline">聊天记录</span>
          </button>
        </div>
      </div>

      <main
        ref={scrollRef}
        className="flex-1 overflow-y-auto w-full flex justify-center py-6 px-4 sm:px-6 scroll-smooth"
      >
        <div className="layout-content-container flex flex-col w-full max-w-[800px] justify-start pb-4">
          {messages.map((msg, idx) => (
            <ChatMessage
              key={idx}
              message={msg.text}
              isAi={msg.isAi}
              sources={msg.sources}
              isThinking={msg.isThinking}
              isStreaming={msg.isStreaming}
            />
          ))}
        </div>
      </main>

      <footer className="bg-background-light/95 dark:bg-background-dark/95 backdrop-blur-md border-t border-slate-200 dark:border-surface-lighter z-20 pb-8 pt-4 w-full">
        <div className="max-w-[800px] mx-auto w-full px-4 sm:px-6">
          <form onSubmit={handleSubmit} className="relative group bg-white dark:bg-surface-lighter rounded-xl shadow-lg border border-slate-200 dark:border-surface-lighter/50 focus-within:ring-2 focus-within:ring-primary/50 focus-within:border-primary transition-all duration-200">
            <div className="flex items-center p-2 gap-2">
              <button type="button" className="flex items-center justify-center w-10 h-10 rounded-lg text-slate-400 hover:text-primary hover:bg-primary/5 transition-colors">
                <span className="material-symbols-outlined">attach_file</span>
              </button>
              <input
                className="flex-1 bg-transparent border-0 focus:ring-0 text-slate-900 dark:text-slate-100 placeholder:text-slate-400 py-3 text-base"
                placeholder="向猪你好运提问任何问题..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary text-white hover:bg-primary/90 shadow-sm disabled:opacity-50 disabled:bg-slate-400 transition-all active:scale-95"
              >
                <span className="material-symbols-outlined">{isLoading ? "hourglass_empty" : "send"}</span>
              </button>
            </div>
          </form>
          <p className="text-center text-[11px] text-slate-400 dark:text-slate-600 mt-3 font-medium">
            猪你好运可能会出错，重要信息请务必自行核实。
          </p>
        </div>
      </footer>
    </div>
  );
}
