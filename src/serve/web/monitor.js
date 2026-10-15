"use strict";
/* Request-monitor page (/api-monitor): reuses the shared app.js helpers
   ($, $_, esc, showBanner, initCommon, QinferRequests). */
(function () {
  let cache = [];
  let activeId = null;

  function renderFiltered() {
    const q = ($("req-search").value || "").toLowerCase();
    const f = $("req-filter").value;
    let list = cache;
    if (q || f) {
      list = list.filter(function (r) {
        const hay = ((r.path || "") + " " + (r.model || "") + " " + (r.state || "") + " " + (r.id || "")).toLowerCase();
        return (!q || hay.indexOf(q) >= 0) && (!f || r.state === f);
      });
    }
    QinferRequests.renderList(list, $("req-list"), function (rec) {
      activeId = rec.id;
      QinferRequests.loadDetail(rec.id, $("req-detail"));
    });
  }

  async function refresh() {
    try {
      const res = await api("/api/requests");
      if (!res.ok) throw new Error("HTTP " + res.status);
      const d = await res.json();
      cache = d.requests || [];
      renderFiltered();
    } catch (e) { /* retry on the next poll */ }
  }

  // Reply feedback captured from the chat (/api/feedback), newest first.
  async function refreshFeedback() {
    try {
      const res = await api("/api/feedback");
      if (!res.ok) return;
      const d = await res.json();
      const el = $("feedback-list");
      if (!el) return;
      const items = d.data || [];
      if (!items.length) {
        el.innerHTML = '<div class="note">' + esc(tr("mon.noFeedback")) + "</div>";
        return;
      }
      el.innerHTML = items.map(function (f) {
        const when = f.at ? new Date(f.at * 1000).toLocaleTimeString() : "";
        return '<div class="feedback-item"><span class="badge">' + esc(f.rating || "") + "</span> " +
          esc(f.excerpt || "") + ' <span class="muted small">' + esc(when) + "</span></div>";
      }).join("");
    } catch (e) { /* retry on the next poll */ }
  }

  document.addEventListener("DOMContentLoaded", function () {
    $("req-search").addEventListener("input", renderFiltered);
    $("req-filter").addEventListener("change", renderFiltered);
    refresh();
    refreshFeedback();
    window.setInterval(function () { refresh(); refreshFeedback(); }, 2000);
  });
})();