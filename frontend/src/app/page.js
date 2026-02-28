"use client";

import { useState, useRef, useEffect } from "react";
import ChatMessage from "@/components/ChatMessage";

export default function ChatPage() {
  const [messages, setMessages] = useState([
    { text: "你好！我是猪你好运，你的智能助手。你可以将文档上传到知识库，然后向我提问。", isAi: true, sources: [] }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg = { text: input, isAi: false };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    const aiMsg = { text: "", isAi: true, sources: [] };
    setMessages(prev => [...prev, aiMsg]);

    try {
      const response = await fetch("http://localhost:8001/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input }),
      });

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
            setMessages(prev => {
              const newMsgs = [...prev];
              const last = { ...newMsgs[newMsgs.length - 1] };
              if (data.token) {
                last.text += data.token;
              } else if (data.sources) {
                last.sources = data.sources;
              }
              newMsgs[newMsgs.length - 1] = last;
              return newMsgs;
            });
          } catch (err) {
            console.error("Error parsing SSE data", err);
          }
        }
      }
    } catch (error) {
      console.error("Chat error:", error);
      setMessages(prev => {
        const newMsgs = [...prev];
        newMsgs[newMsgs.length - 1].text = "抱歉，发生了错误。请检查后端服务是否正常运行。";
        return newMsgs;
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full relative">
      <main
        ref={scrollRef}
        className="flex-1 overflow-y-auto w-full flex justify-center py-6 px-4 sm:px-6 scroll-smooth"
      >
        <div className="layout-content-container flex flex-col w-full max-w-[800px] justify-start pb-4">
          {messages.map((msg, idx) => (
            <ChatMessage key={idx} message={msg.text} isAi={msg.isAi} sources={msg.sources} />
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
