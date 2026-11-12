/* Shared interface behaviour. One definition, so every page behaves the same.
   Light/dark is a selected pair of steps, not an inversion — see shared/ui.css. */
(function(){
  const sysDark = window.matchMedia("(prefers-color-scheme: dark)");
  let stored = null;
  try { stored = localStorage.getItem("ui-theme"); } catch (e) {}
  if (stored) document.documentElement.setAttribute("data-theme", stored);

  const isDark = () => (document.documentElement.getAttribute("data-theme")
    || (sysDark.matches ? "dark" : "light")) === "dark";

  function wire(btn){
    if (!btn) return;
    const sync = () => { btn.textContent = isDark() ? "Light" : "Dark"; };
    btn.addEventListener("click", () => {
      const next = isDark() ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("ui-theme", next); } catch (e) {}
      sync();
      dispatchEvent(new CustomEvent("themechange", {detail: {dark: next === "dark"}}));
    });
    sysDark.addEventListener("change", sync);
    sync();
  }

  const start = () => wire(document.getElementById("themebtn"));
  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", start);
  else start();
})();

/* =====================================================================
   Recording what the resolver decided.

   Every page that shows a suggestion uses these, so "accept" and
   "override" mean the same thing and are measured the same way wherever
   they are clicked.
   ===================================================================== */
window.Decisions = (function(){
  const esc = s => String(s == null ? "" : s).replace(/[&<>"]/g,
    c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

  /* The ticket currently on screen, once the server has given it a reference.
     Every later action on the same ticket sends this back, so a reject, an
     enrichment and a referral are three events on ONE ticket rather than three
     separate tickets. Cleared by newTicket() when the form is scored afresh --
     without that, the next ticket would inherit this one's identity and the
     queue would quietly merge two different problems into one row. */
  let currentRef = null;

  function newTicket(){ currentRef = null; }
  function ticketRef(){ return currentRef; }

  async function send(payload){
    try {
      const r = await fetch("/api/decision", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({...payload, db_ref: currentRef})
      });
      const d = await r.json();
      if (d && d.ticket && d.ticket.ref) currentRef = d.ticket.ref;
      dispatchEvent(new CustomEvent("decision", {detail: d}));
      return d;
    } catch (e) {
      return {error: String(e)};
    }
  }

  /* A confirmation the resolver can actually read: what was suggested, what
     they chose, and whether those differ. */
  function confirmation(kind, suggested, chosen){
    const over = suggested && chosen && suggested !== chosen;
    return `<div class="recorded ${over ? "over" : ""}">
      <span>${over ? "&#9998;" : "&#10003;"}</span>
      <span>${over
        ? `${esc(kind)} <b>overridden</b> &mdash; suggested ${esc(suggested)},
           recorded <b>${esc(chosen)}</b>.`
        : `${esc(kind)} <b>accepted</b> &mdash; recorded <b>${esc(chosen)}</b>.`}
      </span></div>`;
  }

  function tally(s){
    if (!s || !s.total) return `<span class="t">No decisions recorded yet.</span>`;
    const pc = v => v == null ? "&mdash;" : (v * 100).toFixed(0) + "%";
    const bits = [
      `<b>${s.total}</b>decided`,
      `<b>${pc(s.category.accept_rate)}</b>category accepted`,
      `<b>${pc(s.team.accept_rate)}</b>team accepted`,
    ];
    // The measures Section 5.4.2 asks for, shown only once they have happened
    // so an empty session does not read as a row of zeroes.
    if (s.referral_rate != null) bits.push(`<b>${pc(s.referral_rate)}</b>referred on`);
    if (s.abstained) bits.push(`<b>${s.abstained}</b>declined to first line`);
    if (s.closed_as_duplicate) bits.push(`<b>${s.closed_as_duplicate}</b>closed as duplicate`);
    if (s.rejected) bits.push(`<b>${s.rejected}</b>rejected`);
    if (s.mean_confidence_lift != null)
      bits.push(`<b>${s.mean_confidence_lift > 0 ? "+" : ""}${s.mean_confidence_lift.toFixed(3)}</b>` +
                `confidence from detail added (&times;${s.enriched})`);
    if (s.resolutions_reused) bits.push(`<b>${s.resolutions_reused}</b>resolutions reused`);
    return bits.map(b => `<span class="t">${b}</span>`).join("");
  }

  /* One card per distinct fix: what it was, how much stands behind it, and
     how often a fix retrieved at that similarity actually transferred. */
  function fixes(groups, opts){
    if (!groups || !groups.length) return "";
    const o = opts || {};
    return groups.map(g => {
      const a = g.applicability || {};
      const rate = a.observed_transfer_rate
        ? ` &middot; applied in ${(a.observed_transfer_rate * 100).toFixed(1)}% of held-out cases at this similarity`
        : "";
      return `<div class="fix ${g.can_resolve ? "apply" : ""}">
        <div class="fixhead">
          <span class="fixband">${esc(a.band || "")}</span>
          <span class="fixev">resolved ${g.seen_in} past ticket${g.seen_in === 1 ? "" : "s"}
            &middot; closest ${g.similarity.toFixed(3)}</span>
        </div>
        <div class="fixtext">${esc(g.resolution)}</div>
        <div class="fixfoot">
          ${g.can_resolve
            ? `<button class="act primary act-sm" data-resolve="${esc(g.ticket_id)}">Resolve with this</button>`
            : `<button class="act act-sm" data-use="${esc(g.ticket_id)}">Use as reference</button>`}
          <span class="fixsrc">${esc(g.tickets.slice(0, 4).join(", "))}${g.tickets.length > 4 ? "\u2026" : ""}</span>
        </div>
        <div class="hint" style="margin-top:6px">${esc(a.band || "")}${rate}</div>
      </div>`;
    }).join("");
  }

  const summary = () => fetch("/api/decisions").then(r => r.json()).catch(() => null);
  return {send, confirmation, tally, summary, esc, fixes,
          newTicket, ticketRef};
})();
