/* ============================================================
   DERPR PORTAL — mock data, faithful to the engine-adapter contract
   Shapes mirror:
     GET /api/v1/persona/{name}
     GET /api/v1/tools/catalog
     GET /api/v1/session/{persona}/transcript   (build_transcript → chunks)
     GET /api/v1/session/{persona}/ltm_block
   Nothing here is invented past what those endpoints return.
   ============================================================ */

// ---- GET /api/v1/persona/{name} ----------------------------------
const PERSONA = {
  name: "assistant",
  display_name: "Assistant",
  prompt:
    "You are a terse internal IT-support assistant for the DERPR team.\n" +
    "Prefer tool calls over guessing. Never fabricate ticket IDs or cert serials.\n" +
    "Cert/credential resets are sensitive — state the policy before acting.",
  model_name: "gpt-4o-mini",
  // base params (every provider)
  temperature: 0.4,
  max_tokens: 1024,
  history_messages: 24,
  thinking_level: "medium",
  memory_mode: "GLOBAL",          // CHANNEL_ISOLATED | SERVER_WIDE | PERSONAL | GLOBAL | TICKET_ISOLATED
  max_context_tokens: 16384,
  chat_template: "chatml",        // engine-adapter only field
  // kobold-only (provider_extra → kobold), shown only when present
  top_p: 0.92,
  top_k: 40,
  instruct_tags: { system: "<|im_start|>system", user: "<|im_start|>user", assistant: "<|im_start|>assistant" },
  kobold_extras: {
    rep_pen: 1.07, rep_pen_range: 320, rep_pen_slope: 0.7,
    min_p: 0.05, typical: 1.0, tfs: 1.0,
    mirostat: 0, mirostat_tau: 5.0, mirostat_eta: 0.1,
    sampler_order: [6, 0, 1, 3, 4, 2, 5],
  },
  enabled_tools: ["search_tickets", "reset_vpn_cert", "email_user", "lookup_user"],
  tool_policy: { mode: "CONFIRM", confirm_writes: true, auto_read: true },
  security_blocked: false,
  security_block_reasons: [],
};

// ---- GET /api/v1/tools/catalog -----------------------------------
const TOOLS = [
  { name: "search_tickets", description: "Full-text search the Zammad ticket store.",
    is_write: false, capabilities: { locality: "remote", sensitivity: "low", produces_untrusted: true } },
  { name: "lookup_user", description: "Resolve a username to directory record.",
    is_write: false, capabilities: { locality: "local", sensitivity: "low", produces_untrusted: false } },
  { name: "reset_vpn_cert", description: "Reissue a user's VPN client certificate.",
    is_write: true, capabilities: { locality: "local", sensitivity: "high", produces_untrusted: false } },
  { name: "email_user", description: "Send a templated email to a directory user.",
    is_write: true, capabilities: { locality: "remote", sensitivity: "medium", produces_untrusted: false } },
];

// ---- channel tags (source-agnostic `channel` string per the engine) ----
const CHANNELS = [
  { group: "Web UI", items: [
    { id: "web_ui:scratch", name: "assistant · scratch", source: "web", persona: "assistant", active: true, preview: "reissued jdoe's VPN cert and noted…" },
    { id: "web_ui:tune",    name: "gemini · prompt-tune", source: "web", persona: "gemini", preview: "temp 0.4 felt better, keep it" },
  ]},
  { group: "Discord", items: [
    { id: "discord:it",  name: "claude · #it-support", source: "dsc", persona: "claude", preview: "ticket 8821 escalated to L2" },
    { id: "discord:ops", name: "dispatch · #ops",       source: "dsc", persona: "dispatch", preview: "routed 3 notifications" },
  ]},
  { group: "Zammad", items: [
    { id: "zammad:q2", name: "triage · queue#2", source: "zmd", persona: "triage", preview: "stage 2/3 · classifying" },
  ]},
  { group: "Gmail", items: [
    { id: "gmail:sup", name: "support@ inbox", source: "gml", persona: "assistant", preview: "2 unread · vpn access req" },
  ]},
];

// ---- GET /api/v1/session/{persona}/ltm_block ---------------------
// Not a transcript chunk — fetched separately, injected at author's-note slot.
const LTM_BLOCK = {
  count: 2,
  tokens: 2106,
  text:
    "[mem] jdoe = Jane Doe, Engineering. Prefers email confirmations.\n" +
    "[mem] VPN cert resets require L2 approval per policy DP-204.",
};

// ---- token budget (derived; mirrors max_context_tokens accounting) ----
const BUDGET = {
  max: 16384,
  segments: [
    { key: "system",  label: "system prompt", tokens: 1420, color: "var(--accent-dim)" },
    { key: "ltm",     label: "LTM / anote",   tokens: 2106, color: "var(--mem)" },
    { key: "history", label: "history",       tokens: 5030, color: "rgba(150,170,205,0.45)" },
    { key: "reserve", label: "reply reserve", tokens: 1024, color: "var(--write)" },
  ],
};

// ---- GET /api/v1/session/{persona}/transcript  → { chunks: [...] } ----
// Each chunk EXACTLY mirrors build_transcript():
//   interaction_id | role | content | ephemeral | reasoning | tool_context | has_versions
// (+ ephemeral_chunk_id on the parked confirmation). reasoning is folded into
// content as <think>…</think> by the server; we keep the raw `reasoning` too.
const TRANSCRIPT = { chunks: [
  {
    interaction_id: 1041, role: "user",
    content: "Can you reset the VPN cert for jdoe and confirm it on the ticket?",
    ephemeral: false, reasoning: null, tool_context: null, has_versions: false,
  },
  {
    interaction_id: 1042, role: "assistant",
    content: "<think>\nPolicy DP-204 requires L2 approval for cert resets. The tool is is_write+high sensitivity, persona tool_policy is CONFIRM, so this must park for approval before running.\n</think>\nHeads up — reissuing a VPN cert is gated by policy DP-204 (L2 approval). I've staged the reset; approve below and I'll run it.",
    ephemeral: false,
    reasoning: "Policy DP-204 requires L2 approval for cert resets. The tool is is_write+high sensitivity, persona tool_policy is CONFIRM, so this must park for approval before running.",
    tool_context: null, has_versions: true,
  },
  {
    interaction_id: 1044, role: "assistant",
    content: "Done — reissued jdoe's VPN cert and noted it on ticket #8821.",
    ephemeral: false, reasoning: null,
    tool_context: [
      { call_id: "c_91a2", group_id: "g_1", tool_name: "reset_vpn_cert",
        arguments: { user: "jdoe", reason: "ticket #8821" },
        result: "cert reissued · serial 0x4F…A1 · expires 2027-06-03", error: null },
    ],
    has_versions: false,
  },
  {
    interaction_id: 1045, role: "user",
    content: "Great. Email Jane the confirmation too.",
    ephemeral: false, reasoning: null, tool_context: null, has_versions: false,
  },
  // ---- live parked confirmation (CONFIRM mode). Resolved by the NEXT turn,
  //      not a dedicated endpoint. interaction_id=null + ephemeral=true. ----
  {
    interaction_id: null, ephemeral_chunk_id: "pend_7f3c",
    role: "assistant",
    content: "About to run email_user(to=\"jdoe@corp\", template=\"cert_confirm\"). This is a write — approve to send.",
    ephemeral: true, reasoning: null,
    tool_context: [
      { call_id: "c_b4e0", group_id: "g_2", tool_name: "email_user",
        arguments: { to: "jdoe@corp", template: "cert_confirm", ticket: "8821" },
        result: null, error: null },
    ],
    has_versions: false,
  },
]};

// version stack for the one chunk with has_versions=true
// (GET /interaction/1042/versions — canonical last)
const VERSIONS_1042 = [
  { version: 1, content: "Resetting now.", canonical: false },
  { version: 2, content: "I can reset it, but DP-204 wants L2 sign-off first. Proceed?", canonical: false },
  { version: 3, content: "Heads up — reissuing a VPN cert is gated by policy DP-204 (L2 approval). I've staged the reset; approve below and I'll run it.", canonical: true },
];

// ---- CLIENT-SIDE DEV MESSAGES (ephemeral, NOT in the transcript contract) ----
// Dev commands (`/set temp 0.4`, `/detail`, etc.) short-circuit before any DB
// logging — the engine's `preprocess_message` handles them and returns a response
// via DoneEvent(response_type=DEV_COMMAND). These are client-side records only:
// they appear inline in the transcript as thin, collapsible rows but are not
// persisted and vanish on refresh. The `/dev_command` REST endpoint returns
// { response, mutated }; the SSE path emits a DevCommandText delta + a `derpr`
// id-frame with response_type="DEV_COMMAND".
const DEV_MESSAGES = [
  { id: 1, command: "/set temp 0.4", response: "Temperature set to 0.40", mutated: true,
    timestamp: "2026-06-05T14:01:12Z", afterChunkId: 1041 },
  { id: 2, command: "/detail", response: "assistant — model: gpt-4o-mini · temp: 0.40 · top_p: 0.92 · top_k: 40 · max_tokens: 1024 · history: 24 · memory: GLOBAL · thinking: medium · tools: CONFIRM (4 enabled)",
    mutated: false, timestamp: "2026-06-05T14:03:45Z", afterChunkId: 1044 },
];

window.DERPR = { PERSONA, TOOLS, CHANNELS, LTM_BLOCK, BUDGET, TRANSCRIPT, VERSIONS_1042, DEV_MESSAGES };

// ---- proposed: GET /api/v1/session/{persona}/assemble?message=... (DRY RUN) ----
// The anti-divergence primitive. Returns the EXACT request the engine would
// send — produced by running chat_system.stream_response's assembler in a
// no-inference mode, NOT reconstructed client-side. Same code path as a live
// submit, so the inspector can never drift from what actually hits the model.
// parity.source === "engine.dry_run" asserts this; if a build ever falls back
// to client reconstruction it flips to "client_fallback" and the UI warns.
window.DERPR.ASSEMBLED_REQUEST = {
  parity: { source: "engine.dry_run", builder: "chat_system.stream_response", matches_live: true },
  route: "engine · POST /v1/chat/completions",
  model_name: "gpt-4o-mini",
  // resolved sampling params actually forwarded (local_inference_config). Base
  // persona props + any kobold sampler extras the engine extracts for the route.
  params: {
    temperature: 0.4, top_p: 0.92, top_k: 40, max_tokens: 1024, stop: null,
    // extras pulled through into local_inference_config by the engine adapter:
    rep_pen: 1.07, min_p: 0.05, tfs: 1.0,
  },
  // history rebuilt from DB (client_messages discarded). Each entry tags the
  // interaction_id it came from so a row in the transcript maps to a wire line.
  messages: [
    { role: "system", content: "You are a terse internal IT-support assistant for the DERPR team.\nPrefer tool calls over guessing. Never fabricate ticket IDs or cert serials.\nCert/credential resets are sensitive — state the policy before acting.", src: "persona.prompt" },
    { role: "system", content: "[author's-note]\n[mem] jdoe = Jane Doe, Engineering. Prefers email confirmations.\n[mem] VPN cert resets require L2 approval per policy DP-204.", src: "ltm_block" },
    { role: "user", content: "Can you reset the VPN cert for jdoe and confirm it on the ticket?", src: "#1041" },
    { role: "assistant", content: "Heads up — reissuing a VPN cert is gated by policy DP-204 (L2 approval). I've staged the reset; approve below and I'll run it.", src: "#1042 · v3 canonical" },
    { role: "assistant", content: "Done — reissued jdoe's VPN cert and noted it on ticket #8821.", src: "#1044", tool_note: "tool result folded by engine" },
    { role: "user", content: "Great. Email Jane the confirmation too.", src: "#1045" },
  ],
};
