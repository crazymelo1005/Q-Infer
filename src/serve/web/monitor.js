"use strict";
/* 请求检查页（/api-monitor）：复用 app.js 的 $_api_esc_showBanner_initCommon_QinferRequests。 */
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
    } catch (e) { /* 下一次轮询再试 */ }
  }

  document.addEventListener("DOMContentLoaded", function () {
    $("req-search").addEventListener("input", renderFiltered);
    $("req-filter").addEventListener("change", renderFiltered);
    refresh();
    window.setInterval(refresh, 2000);
  });
})();