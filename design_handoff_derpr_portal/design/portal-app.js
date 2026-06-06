/* ============================================================
   DERPR PORTAL — Control Room app logic (vanilla, contract-driven)
   Renders the conversation column from DERPR.TRANSCRIPT.chunks, the
   exact shape build_transcript() returns. RENDERED view stitches in
   the persona system prompt + LTM block (separate fetches); CONTEXT
   view shows the assembled prompt as it goes to the model.
   ============================================================ */
(function () {
  const D = window.DERPR;
  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const esc = (s) => (s || "").replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const fmtTok = (n) => n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n);

  let viewMode = "rendered";       // 'rendered' | 'context'
  let ltmOn = true;
  let chunks = D.TRANSCRIPT.chunks.map(c => ({ ...c })); // working copy
  let canonical1042 = 3;           // which version index is canonical

  // Dev messages — client-side ephemeral state. These are not part of the
  // DP-130 transcript contract; they record dev command input/response pairs
  // that the engine never persists. Rendered inline as thin collapsible rows.
  let devMessages = (D.DEV_MESSAGES || []).map(d => ({ ...d }));
  let devMsgIdCounter = devMessages.length;

  // strip a folded <think> block out of content → { reasoning, body }
  function splitThink(content) {
    const m = content.match(/^<think>\n([\s\S]*?)\n<\/think>\n([\s\S]*)$/);
    if (m) return { reasoning: m[1], body: m[2] };
    return { reasoning: null, body: content };
  }

  // ---- DEV MESSAGE ROW (thin, collapsible, ephemeral) ----
  function renderDevMsg(dm) {
    const row = el("div", "devrow");
    const ts = dm.timestamp ? new Date(dm.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
    // Truncate response preview to ~60 chars for collapsed view
    const preview = (dm.response || "").length > 60
      ? dm.response.substring(0, 60) + "…"
      : dm.response;
    row.innerHTML = `
      <div class="dev-header">
        <span class="dev-label">dev</span>
        <span class="dev-cmd">${esc(dm.command)}</span>
        <span class="dev-arrow">→</span>
        <span class="dev-resp-preview">${esc(preview)}</span>
        ${dm.mutated ? '<span class="dev-mutated">mutated</span>' : ''}
        <span class="dev-ts">${esc(ts)}</span>
        <span class="dev-car">▸</span>
      </div>
      <div class="dev-body">
        <div class="dev-full-resp">${esc(dm.response)}</div>
        <div class="dev-ephemeral-note">ephemeral · not persisted · vanishes on refresh</div>
      </div>`;
    row.addEventListener("click", () => row.classList.toggle("open"));
    return row;
  }

  // ---- TOOL CARD ----
  function toolCard(tc) {
    const def = D.TOOLS.find(t => t.name === tc.tool_name) || { is_write: false, capabilities: {} };
    const caps = def.capabilities || {};
    const card = el("div", "tool" + (def.is_write ? "" : " read"));
    const sens = caps.sensitivity;
    const badges = [
      def.is_write ? `<span class="badge write">write</span>` : `<span class="badge read">read</span>`,
      sens === "high" ? `<span class="badge high">sensitive</span>` : (sens === "medium" ? `<span class="badge med">medium</span>` : ""),
      caps.locality ? `<span class="badge ${caps.locality}">${caps.locality}</span>` : "",
      caps.produces_untrusted ? `<span class="badge low">untrusted out</span>` : "",
    ].join("");
    const argStr = Object.entries(tc.arguments || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
    card.innerHTML = `
      <button class="th">
        <span class="fn">→ <b>${esc(tc.tool_name)}</b>(${esc(argStr)})</span>
        ${badges}
        <span class="car">▸</span>
      </button>
      <div class="tb">
        <div class="kv"><span class="k">call_id</span><span class="v">${esc(tc.call_id)}${tc.group_id ? " · " + esc(tc.group_id) : ""}</span></div>
        <div class="kv"><span class="k">args</span><span class="v">${esc(JSON.stringify(tc.arguments))}</span></div>
        ${tc.result != null
          ? `<div class="result ok">${esc(tc.result)}</div>`
          : (tc.error ? `<div class="result" style="color:var(--danger)">${esc(tc.error)}</div>`
                      : `<div class="result pending">awaiting approval — tool not yet run</div>`)}
      </div>`;
    card.querySelector(".th").addEventListener("click", () => card.classList.toggle("open"));
    return card;
  }

  // ---- MESSAGE ROW (rendered view) ----
  function renderMsg(c) {
    const { reasoning, body } = splitThink(c.content);
    const row = el("div", "msg" + (c.ephemeral ? " ephemeral" : ""));
    const idText = c.ephemeral
      ? `<span class="idtag eph">ephemeral · ${esc(c.ephemeral_chunk_id || "pending")}</span>`
      : (c.interaction_id != null ? `<span class="idtag">#${c.interaction_id}</span>` : `<span class="idtag">unaddressable</span>`);

    // version chevrons ONLY when has_versions
    let chev = "";
    if (c.has_versions && c.interaction_id === 1042) {
      const total = D.VERSIONS_1042.length;
      chev = `<span class="chev"><button data-vprev>‹</button><span class="ct">${canonical1042}&#8202;/&#8202;${total}</span><button data-vnext>›</button></span><span class="lbl">version</span>`;
    }

    const who = c.role === "assistant" ? "assistant" : "portal";
    row.innerHTML = `
      <div class="gut"><div class="av ${c.role}">${c.role === "assistant" ? "AS" : "U"}</div></div>
      <div class="bd">
        <div class="meta">
          <span class="who ${c.role}">${who}</span>
          <span class="ts">14:0${(c.interaction_id || 7) % 9}</span>
          ${idText}
          ${chev}
          ${c.ephemeral ? `<span class="chip" style="color:var(--write);border-color:rgba(231,173,98,.4)"><span class="dot" style="background:var(--write)"></span>awaiting approval</span>` : ""}
        </div>
      </div>`;
    const bd = row.querySelector(".bd");

    // reasoning fold
    if (reasoning) {
      const fold = el("div", "fold");
      fold.innerHTML = `
        <button class="fh"><span class="tw">⟁ reasoning</span><span style="color:var(--ink-faint)">${fmtTok(Math.round(reasoning.length / 3.6))} tok</span><span class="car">▸</span></button>
        <div class="fb"><div class="t">${esc(reasoning)}</div></div>`;
      fold.querySelector(".fh").addEventListener("click", () => fold.classList.toggle("open"));
      bd.appendChild(fold);
    }

    // tool calls
    (c.tool_context || []).forEach(tc => bd.appendChild(toolCard(tc)));

    // body text
    if (body && body.trim()) {
      const t = el("div", "text");
      t.textContent = body;
      t.style.marginTop = (reasoning || (c.tool_context || []).length) ? "10px" : "0";
      bd.appendChild(t);
    }

    // CONFIRM bar for ephemeral parked write
    if (c.ephemeral) {
      const cf = el("div", "confirm");
      cf.innerHTML = `
        <span class="lbl">CONFIRM</span>
        <button class="btn approve" data-approve>✓ approve &amp; run</button>
        <button class="btn deny" data-deny>✕ deny</button>
        <button class="btn" data-editargs>edit args</button>
        <span class="note">resolves via next /chat/completions turn · persona tool_policy = CONFIRM</span>`;
      cf.querySelector("[data-approve]").addEventListener("click", () => resolveConfirm(c, true));
      cf.querySelector("[data-deny]").addEventListener("click", () => resolveConfirm(c, false));
      bd.appendChild(cf);
    }

    // row actions (not on ephemeral)
    if (!c.ephemeral) {
      const acts = el("div", "rowacts");
      acts.innerHTML = c.role === "assistant"
        ? `<button class="ract" title="regenerate">⟲ regen</button><button class="ract" title="edit">✎ edit</button><button class="ract danger" title="suppress">✕ del</button>`
        : `<button class="ract" title="edit">✎ edit</button><button class="ract danger" title="suppress">✕ del</button>`;
      row.appendChild(acts);
    }

    // chevron wiring
    const prev = row.querySelector("[data-vprev]"), next = row.querySelector("[data-vnext]");
    if (prev) prev.addEventListener("click", () => swapVersion(-1));
    if (next) next.addEventListener("click", () => swapVersion(1));

    return row;
  }

  function swapVersion(dir) {
    const total = D.VERSIONS_1042.length;
    canonical1042 = ((canonical1042 - 1 + dir + total) % total) + 1;
    const c = chunks.find(x => x.interaction_id === 1042);
    if (c) {
      const v = D.VERSIONS_1042[canonical1042 - 1];
      // keep the think block, swap the body
      const { reasoning } = splitThink(c.content);
      c.content = reasoning ? `<think>\n${reasoning}\n</think>\n${v.content}` : v.content;
    }
    paint();
  }

  function resolveConfirm(c, approved) {
    const idx = chunks.indexOf(c);
    if (idx < 0) return;
    if (approved) {
      const tc = (c.tool_context || [])[0] || {};
      chunks[idx] = {
        interaction_id: 1047, role: "assistant",
        content: `Sent — emailed jdoe@corp the cert confirmation for ticket #8821.`,
        ephemeral: false, reasoning: null,
        tool_context: [{ ...tc, result: "202 accepted · message id <a91f@mx.corp>" }],
        has_versions: false,
      };
    } else {
      chunks[idx] = {
        interaction_id: 1047, role: "assistant",
        content: `Cancelled — did not send the email. Let me know if you'd like to revise it.`,
        ephemeral: false, reasoning: null, tool_context: null, has_versions: false,
      };
    }
    paint();
  }

  // ---- CONTEXT VIEW (assembled prompt) ----
  function renderContext() {
    const frag = document.createDocumentFragment();
    const note = el("div", "ctxnote");
    note.innerHTML = `<b>Assembled exactly as sent to the model</b> — rebuilt by the engine's history-gathering code (same path as a real request). System + author's-note are not transcript chunks; they're stitched here.`;
    frag.appendChild(note);

    // system
    frag.appendChild(ctxRow("system", "⟦system⟧", D.PERSONA.prompt, 1420));
    // author's-note / LTM
    if (ltmOn) frag.appendChild(ctxRow("anote", "⟦author's-note · LTM⟧", D.LTM_BLOCK.text, D.LTM_BLOCK.tokens));
    // visible turns (skip ephemeral parked confirmation — not yet in prompt)
    chunks.filter(c => !c.ephemeral).forEach(c => {
      const { body } = splitThink(c.content);
      const tools = (c.tool_context || []).map(t => `  → ${t.tool_name}(${Object.entries(t.arguments).map(([k, v]) => k + "=" + JSON.stringify(v)).join(", ")}) ⇒ ${t.result || "—"}`).join("\n");
      const text = tools ? (body ? body + "\n" + tools : tools) : body;
      frag.appendChild(ctxRow(c.role, `⟦${c.role}⟧`, text, Math.round((text || "").length / 3.6)));
    });
    return frag;
  }
  function ctxRow(cls, role, text, tok) {
    const r = el("div", "ctxrow " + cls);
    r.innerHTML = `<div class="lh"><span class="role">${esc(role)}</span><span class="tcount">${fmtTok(tok)} tok</span></div>`;
    const t = el("div", "text"); t.textContent = text; r.appendChild(t);
    return r;
  }

  // ---- PAINT TRANSCRIPT ----
  function paint() {
    const wrap = $("#transcript");
    wrap.innerHTML = "";
    if (viewMode === "rendered") {
      // pinned system prompt
      const sys = el("div", "sysrow");
      sys.innerHTML = `<div class="lh"><span class="lbl">System · persona prompt</span><button class="mini" id="editsys">edit ✎</button><span class="lbl" style="margin-left:auto">GET /persona/${D.PERSONA.name}</span></div><div class="text">${esc(D.PERSONA.prompt)}</div>`;
      wrap.appendChild(sys);

      chunks.forEach((c, i) => {
        // inject LTM right after the first user turn, in rendered view
        if (ltmOn && i === 1) {
          const ltm = el("div", "ltmrow");
          ltm.innerHTML = `<div class="lh"><span class="lbl">◈ LTM recalled · injected as author's-note</span><span class="chip mem"><span class="dot"></span>${D.LTM_BLOCK.count} memories · ${fmtTok(D.LTM_BLOCK.tokens)} tok</span><span class="lbl" style="margin-left:auto">/session/${D.PERSONA.name}/ltm_block</span></div><div class="text">${D.LTM_BLOCK.text.replace(/\[mem\]/g, '<span class="m">[mem]</span>')}</div>`;
          wrap.appendChild(ltm);
        }
        wrap.appendChild(renderMsg(c));

        // Interleave any dev messages positioned after this chunk
        const cid = c.interaction_id;
        devMessages.filter(dm => dm.afterChunkId === cid).forEach(dm => {
          wrap.appendChild(renderDevMsg(dm));
        });
      });

      // Dev messages not anchored to any chunk (e.g. sent before any chunks,
      // or after the last chunk) — append at the end
      devMessages.filter(dm => !dm.afterChunkId || !chunks.some(c => c.interaction_id === dm.afterChunkId)).forEach(dm => {
        wrap.appendChild(renderDevMsg(dm));
      });
    } else {
      // Context view — dev messages are NOT part of the LLM prompt, skip them
      wrap.appendChild(renderContext());
    }
    paintBudget();
  }

  // ---- BUDGET ----
  function paintBudget() {
    const bar = $("#budgetbar"), legend = $("#budgetlegend");
    bar.innerHTML = ""; legend.innerHTML = "";
    const segs = D.BUDGET.segments.filter(s => s.key !== "ltm" || ltmOn);
    const used = segs.reduce((a, s) => a + s.tokens, 0);
    segs.forEach(s => {
      const i = el("i"); i.style.background = s.color; i.style.width = (s.tokens / D.BUDGET.max * 100) + "%"; bar.appendChild(i);
      const sp = el("span", null, `<i style="background:${s.color}"></i>${s.label} ${fmtTok(s.tokens)}`); legend.appendChild(sp);
    });
    const total = el("span", "total", `${fmtTok(used)} / ${fmtTok(D.BUDGET.max)} ctx`); legend.appendChild(total);
  }

  // ---- INSPECTOR ----
  function buildInspector() {
    const p = D.PERSONA;
    // persona pane
    const pp = $("#pane-persona");
    pp.innerHTML = `
      <div class="secbanner">⚠ persona security-blocked — ${esc((p.security_block_reasons || []).join("; ") || "tooling disabled")}</div>
      <div class="field"><span class="lbl">Identity</span><div class="ctrl"><span>${esc(p.name)}</span><span style="color:var(--ink-faint)">fork ⑂ · save as…</span></div></div>
      <div class="field"><span class="lbl">System prompt</span><div class="ctrl area">${esc(p.prompt)}</div></div>

      <div class="section">▣ base params<span class="desc">sent to every provider</span></div>
      <div class="field"><div class="row2">
        <div><span class="lbl">model_name</span><div class="ctrl"><span>${esc(p.model_name)}</span><span class="car">▾</span></div></div>
        <div><span class="lbl">memory_mode</span><div class="ctrl"><span>${esc(p.memory_mode)}</span><span class="car">▾</span></div></div>
      </div></div>
      <div class="field"><span class="lbl">temperature · <span id="tval">${p.temperature.toFixed(2)}</span></span>
        <div class="slider"><input type="range" min="0" max="2" step="0.05" value="${p.temperature}" id="temp"><span class="val" id="tval2">${p.temperature.toFixed(2)}</span></div></div>
      <div class="field"><div class="row2">
        <div><span class="lbl">max_tokens</span><div class="ctrl"><span>${p.max_tokens}</span></div></div>
        <div><span class="lbl">history_messages</span><div class="ctrl"><span>${p.history_messages}</span></div></div>
      </div></div>
      <div class="field"><div class="row2">
        <div><span class="lbl">max_context_tokens</span><div class="ctrl"><span>${p.max_context_tokens}</span></div></div>
        <div><span class="lbl">thinking_level</span><div class="ctrl"><span>${esc(p.thinking_level)}</span><span class="car">▾</span></div></div>
      </div></div>
      <div class="field"><div class="row2">
        <div><span class="lbl">chat_template</span><div class="ctrl"><span>${esc(p.chat_template)}</span><span class="car">▾</span></div></div>
        <div><span class="lbl">tool_policy.mode</span><div class="ctrl"><span>${esc(p.tool_policy.mode)}</span><span class="car">▾</span></div></div>
      </div></div>

      <div class="section kobold" id="kobold-sec">⚠ kobold-only<span class="pill">passthrough route</span><span class="desc">provider_extra · only on kcpp endpoint</span><span class="car">▾</span></div>
      <div class="kobold-fields" id="kobold-fields">
        <div class="field"><div class="row2">
          <div><span class="lbl">top_p</span><div class="ctrl"><span>${p.top_p}</span></div></div>
          <div><span class="lbl">top_k</span><div class="ctrl"><span>${p.top_k}</span></div></div>
        </div></div>
        <div class="field"><div class="row2">
          <div><span class="lbl">rep_pen</span><div class="ctrl"><span>${p.kobold_extras.rep_pen}</span></div></div>
          <div><span class="lbl">rep_pen_range</span><div class="ctrl"><span>${p.kobold_extras.rep_pen_range}</span></div></div>
        </div></div>
        <div class="field"><div class="row2">
          <div><span class="lbl">min_p</span><div class="ctrl"><span>${p.kobold_extras.min_p}</span></div></div>
          <div><span class="lbl">tfs</span><div class="ctrl"><span>${p.kobold_extras.tfs}</span></div></div>
        </div></div>
        <div class="field"><span class="lbl">mirostat · tau · eta</span><div class="ctrl"><span>${p.kobold_extras.mirostat} · ${p.kobold_extras.mirostat_tau} · ${p.kobold_extras.mirostat_eta}</span></div></div>
        <div class="field"><span class="lbl">instruct_tags</span><div class="ctrl"><span>${esc(Object.keys(p.instruct_tags).length ? "ChatML (custom)" : "—")}</span><span class="car">▾</span></div></div>
        <div class="field"><span class="lbl">sampler_order</span><div class="ctrl"><span>[${p.kobold_extras.sampler_order.join(", ")}]</span></div></div>
      </div>`;
    // temp slider live
    const temp = pp.querySelector("#temp");
    temp.addEventListener("input", () => { pp.querySelector("#tval").textContent = (+temp.value).toFixed(2); pp.querySelector("#tval2").textContent = (+temp.value).toFixed(2); });
    // kobold collapse
    pp.querySelector("#kobold-sec").addEventListener("click", () => {
      pp.querySelector("#kobold-sec").classList.toggle("collapsed");
      pp.querySelector("#kobold-fields").classList.toggle("collapsed");
    });

    // tools pane
    const tp = $("#pane-tools");
    tp.innerHTML = D.TOOLS.map(t => {
      const caps = t.capabilities;
      const on = D.PERSONA.enabled_tools.includes(t.name);
      const tags = [
        t.is_write ? `<span class="badge write">write</span>` : `<span class="badge read">read</span>`,
        caps.sensitivity === "high" ? `<span class="badge high">high</span>` : (caps.sensitivity === "medium" ? `<span class="badge med">med</span>` : `<span class="badge low">low</span>`),
        caps.locality ? `<span class="badge ${caps.locality}">${caps.locality}</span>` : "",
        caps.produces_untrusted ? `<span class="badge low">untrusted</span>` : "",
      ].join("");
      return `<div class="toolrow">
        <div class="tt"><span class="nm">${esc(t.name)}</span><button class="en ${on ? "on" : ""}" data-tool="${esc(t.name)}"><span class="sw"></span></button></div>
        <div class="ds">${esc(t.description)}</div>
        <div class="tags">${tags}</div>
      </div>`;
    }).join("");
    tp.querySelectorAll(".en").forEach(b => b.addEventListener("click", () => b.classList.toggle("on")));

    // raw request pane — the assembled request, dry-run from the engine builder
    const rp = $("#pane-raw");
    rp.innerHTML = "";
    const A = D.ASSEMBLED_REQUEST;
    const ok = A.parity.source === "engine.dry_run" && A.parity.matches_live;
    const banner = el("div", "parity " + (ok ? "ok" : "warn"));
    banner.innerHTML = ok
      ? `<span class="pi">✓ parity verified</span><span class="pd">dry-run of <code>${esc(A.parity.builder)}</code> — same code path as a live submit. Not reconstructed client-side.</span>`
      : `<span class="pi warn">⚠ client fallback</span><span class="pd">engine dry-run unavailable — request rebuilt in the browser and may drift. Treat as approximate.</span>`;
    rp.appendChild(banner);

    // routing + params
    const head = el("div", "rawsec");
    head.innerHTML = `
      <div class="rk">route</div><div class="rv accent">${esc(A.route)}</div>
      <div class="rk">model_name</div><div class="rv">${esc(A.model_name)}</div>`;
    rp.appendChild(head);

    const psec = el("div", "rawsec");
    psec.innerHTML = `<div class="rawlbl">local_inference_config <span class="rawnote">resolved params forwarded to the provider</span></div>` +
      Object.entries(A.params).map(([k, v]) =>
        `<div class="rk">${esc(k)}</div><div class="rv ${v == null ? "null" : ""}">${v == null ? "null" : esc(String(v))}</div>`
      ).join("");
    rp.appendChild(psec);

    // messages array
    const mhead = el("div", "rawlbl wrap", `messages[] <span class="rawnote">history rebuilt from DB · client array discarded</span>`);
    rp.appendChild(mhead);
    A.messages.forEach((m, i) => {
      const r = el("div", "wire " + m.role);
      r.innerHTML = `
        <div class="wh"><span class="wrole">${esc(m.role)}</span><span class="wsrc">${esc(m.src)}</span><span class="widx">[${i}]</span></div>
        <div class="wtext">${esc(m.content)}</div>
        ${m.tool_note ? `<div class="wnote">↳ ${esc(m.tool_note)}</div>` : ""}`;
      rp.appendChild(r);
    });
    const foot = el("div", "rawfoot", `Edit any line by editing its source row in the transcript — never a free-text blob. The next submit re-runs this exact assembly.`);
    rp.appendChild(foot);
  }

  // ---- WIRE CHROME ----
  function wire() {
    // collapse buttons
    $("#tg-rail").addEventListener("click", () => toggleCol("no-rail", "#tg-rail"));
    $("#tg-chan").addEventListener("click", () => toggleCol("no-chan", "#tg-chan"));
    $("#tg-insp").addEventListener("click", () => toggleCol("no-insp", "#tg-insp"));

    // view segmented
    $("#seg-rendered").addEventListener("click", () => setView("rendered"));
    $("#seg-context").addEventListener("click", () => setView("context"));

    // LTM toggle
    $("#ltm-toggle").addEventListener("click", () => {
      ltmOn = !ltmOn;
      $("#ltm-toggle").classList.toggle("on", ltmOn);
      paint();
    });

    // inspector tabs
    document.querySelectorAll(".insp-tab").forEach(t => t.addEventListener("click", () => {
      document.querySelectorAll(".insp-tab").forEach(x => x.classList.toggle("active", x === t));
      document.querySelectorAll(".insp-pane").forEach(p => p.classList.toggle("active", p.id === "pane-" + t.dataset.pane));
    }));

    // channel select (cosmetic)
    document.querySelectorAll(".chanitem").forEach(c => c.addEventListener("click", () => {
      document.querySelectorAll(".chanitem").forEach(x => x.classList.remove("active"));
      c.classList.add("active");
    }));

    // ---- COMPOSER: dev command routing ----
    // A leading `/` routes to POST /persona/{name}/dev_command instead of chat.
    // In this prototype the response is mocked; in production it would POST to
    // the engine adapter and capture the response.
    const textarea = $("textarea");
    const sendBtn = $(".send");
    if (textarea && sendBtn) {
      const handleSend = () => {
        const text = (textarea.value || "").trim();
        if (!text) return;
        if (text.startsWith("/")) {
          // Dev command — in production: POST /api/v1/persona/{name}/dev_command
          // Here we mock the response for prototype demonstration.
          const lastChunk = chunks.length > 0 ? chunks[chunks.length - 1] : null;
          devMessages.push({
            id: ++devMsgIdCounter,
            command: text,
            response: `(mock) dev command acknowledged: ${text}`,
            mutated: text.toLowerCase().includes("set"),
            timestamp: new Date().toISOString(),
            afterChunkId: lastChunk ? lastChunk.interaction_id : null,
          });
          textarea.value = "";
          paint();
          // scroll to bottom
          const scroll = $(".scroll");
          if (scroll) setTimeout(() => scroll.scrollTop = scroll.scrollHeight, 50);
        }
        // Non-dev messages: in production would POST to /v1/chat/completions
        // (not implemented in this static prototype)
      };
      sendBtn.addEventListener("click", handleSend);
      textarea.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          handleSend();
        }
      });
    }

    // budget visibility follows view (hide in context — it's shown per-row)
  }
  function toggleCol(cls, btn) {
    const on = document.body.classList.toggle(cls);
    $(btn).setAttribute("aria-pressed", on ? "false" : "true");
  }
  function setView(v) {
    viewMode = v;
    $("#seg-rendered").classList.toggle("on", v === "rendered");
    $("#seg-context").classList.toggle("on", v === "context");
    $("#budget").classList.toggle("hidden", v === "context");
    paint();
  }

  // ---- BUILD CHANNELS ----
  function buildChannels() {
    const list = $("#chanlist");
    D.CHANNELS.forEach(g => {
      list.appendChild(el("div", "changroup", g.group));
      g.items.forEach(it => {
        const b = el("button", "chanitem" + (it.active ? " active" : ""));
        b.innerHTML = `<div class="av">${it.name.slice(0, 2).toUpperCase()}</div>
          <div class="ci"><div class="top"><span class="nm">${esc(it.name)}</span><span class="src ${it.source}">${it.source}</span></div><div class="pv">${esc(it.preview)}</div></div>`;
        list.appendChild(b);
      });
    });
    const nb = el("button", "newchan", "+ new web_ui channel");
    list.appendChild(nb);
  }

  // ---- INIT ----
  document.addEventListener("DOMContentLoaded", () => {
    if (D.PERSONA.security_blocked) document.body.classList.add("persona-blocked");
    buildChannels();
    buildInspector();
    wire();
    paint();
  });
})();
