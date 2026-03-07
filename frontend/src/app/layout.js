import "./globals.css";
import Header from "@/components/Header";

export default function RootLayout({ children }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-background-light dark:bg-background-dark font-display text-slate-900 dark:text-slate-100 flex flex-col h-screen overflow-hidden">
        <Header />
        <main className="flex-1 overflow-hidden relative">
          {children}
        </main>
      </body>
    </html>
  );
}
