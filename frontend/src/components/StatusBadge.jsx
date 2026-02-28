export default function StatusBadge({ status }) {
    const styles = {
        ready: "bg-emerald-500/10 text-emerald-500 border-emerald-500/20",
        processing: "bg-amber-500/10 text-amber-500 border-amber-500/20 animate-pulse",
        error: "bg-rose-500/10 text-rose-500 border-rose-500/20",
    };

    const currentStyle = styles[status] || styles.ready;

    return (
        <div className={`px-2.5 py-0.5 rounded-full border text-[11px] font-bold uppercase tracking-wider ${currentStyle}`}>
            {status}
        </div>
    );
}
