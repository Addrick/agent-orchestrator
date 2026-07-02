/* ============================================================
   Mock fixtures — mirror design/portal-data.js EXACTLY.
   Used by the API client as a fallback when no live engine is
   reachable (build/dev without a backend). The live client hits
   the real endpoints first and only falls back to these.
   ============================================================ */
import type {
  Persona,
  ToolDef,
  ChannelGroup,
  Chunk,
  VersionsResponse,
  AssembledRequest,
} from '../types/contracts'

export const MOCK_PERSONA: Persona = {
  name: 'assistant',
  display_name: 'Assistant',
  prompt:
    'You are a terse internal IT-support assistant for the DERPR team.\n' +
    'Prefer tool calls over guessing. Never fabricate ticket IDs or cert serials.\n' +
    'Cert/credential resets are sensitive — state the policy before acting.',
  model_name: 'gpt-4o-mini',
  temperature: 0.4,
  max_tokens: 1024,
  history_messages: 24,
  thinking_level: 'medium',
  memory_mode: 'GLOBAL',
  long_term_memory: false,
  max_context_tokens: 16384,
  chat_template: 'chatml',
  top_p: 0.92,
  top_k: 40,
  instruct_tags: {
    system: '<|im_start|>system',
    user: '<|im_start|>user',
    assistant: '<|im_start|>assistant',
  },
  kobold_extras: {
    rep_pen: 1.07,
    rep_pen_range: 320,
    rep_pen_slope: 0.7,
    min_p: 0.05,
    typical: 1.0,
    tfs: 1.0,
    mirostat: 0,
    mirostat_tau: 5.0,
    mirostat_eta: 0.1,
    sampler_order: [6, 0, 1, 3, 4, 2, 5],
  },
  enabled_tools: ['search_tickets', 'reset_vpn_cert', 'email_user', 'lookup_user'],
  // realistic engine shape: reads auto-run, writes park for CONFIRM (ask)
  tool_policy: {
    default: 'deny',
    allow: ['search_tickets', 'lookup_user'],
    ask: ['reset_vpn_cert', 'email_user'],
    explicit_overrides: [],
    capabilities_required: [],
  },
  service_bindings: ['zammad'],
  security_blocked: false,
  security_block_reasons: [],
}

export const MOCK_TOOLS: ToolDef[] = [
  {
    name: 'search_tickets',
    description: 'Full-text search the Zammad ticket store.',
    is_write: false,
    service_binding: 'zammad',
    capabilities: { locality: 'remote', sensitivity: 'low', produces_untrusted: true },
  },
  {
    name: 'lookup_user',
    description: 'Resolve a username to directory record.',
    is_write: false,
    service_binding: null,
    capabilities: { locality: 'local', sensitivity: 'low', produces_untrusted: false },
  },
  {
    name: 'reset_vpn_cert',
    description: "Reissue a user's VPN client certificate.",
    is_write: true,
    service_binding: null,
    capabilities: { locality: 'local', sensitivity: 'high', produces_untrusted: false },
  },
  {
    name: 'email_user',
    description: 'Send a templated email to a directory user.',
    is_write: true,
    service_binding: 'zammad',
    capabilities: { locality: 'remote', sensitivity: 'medium', produces_untrusted: false },
  },
]

export const MOCK_CHANNELS: ChannelGroup[] = [
  {
    group: 'Web UI',
    items: [
      {
        id: 'web_ui:scratch',
        channel: 'web_ui',
        source: 'web',
        preview: "reissued jdoe's VPN cert and noted…",
      },
      {
        id: 'web_ui:tune',
        channel: 'web_ui_tune',
        source: 'web',
        preview: 'temp 0.4 felt better, keep it',
      },
    ],
  },
  {
    group: 'Discord',
    items: [
      {
        id: 'discord:it',
        channel: 'discord_it-support',
        source: 'dsc',
        preview: 'ticket 8821 escalated to L2',
      },
      {
        id: 'discord:ops',
        channel: 'discord_ops',
        source: 'dsc',
        preview: 'routed 3 notifications',
      },
    ],
  },
  {
    group: 'Zammad',
    items: [
      {
        id: 'zammad:q2',
        channel: 'zammad',
        source: 'zmd',
        preview: 'stage 2/3 · classifying',
      },
    ],
  },
  {
    group: 'Gmail',
    items: [
      {
        id: 'gmail:sup',
        channel: 'gmail',
        source: 'gml',
        preview: '2 unread · vpn access req',
      },
    ],
  },
]

export const MOCK_LTM_BLOCK =
  '[mem] jdoe = Jane Doe, Engineering. Prefers email confirmations.\n' +
  '[mem] VPN cert resets require L2 approval per policy DP-204.'

export const MOCK_TRANSCRIPT: Chunk[] = [
  {
    interaction_id: 1041,
    role: 'user',
    content: 'Can you reset the VPN cert for jdoe and confirm it on the ticket?',
    ephemeral: false,
    reasoning: null,
    tool_context: null,
    has_versions: false,
  },
  {
    interaction_id: 1042,
    role: 'assistant',
    content:
      '<think>\nPolicy DP-204 requires L2 approval for cert resets. The tool is is_write+high sensitivity, persona tool_policy is CONFIRM, so this must park for approval before running.\n</think>\nHeads up — reissuing a VPN cert is gated by policy DP-204 (L2 approval). I\'ve staged the reset; approve below and I\'ll run it.',
    ephemeral: false,
    reasoning:
      'Policy DP-204 requires L2 approval for cert resets. The tool is is_write+high sensitivity, persona tool_policy is CONFIRM, so this must park for approval before running.',
    tool_context: null,
    has_versions: true,
  },
  {
    interaction_id: 1044,
    role: 'assistant',
    content: "Done — reissued jdoe's VPN cert and noted it on ticket #8821.",
    ephemeral: false,
    reasoning: null,
    tool_context: [
      {
        call_id: 'c_91a2',
        group_id: 'g_1',
        tool_name: 'reset_vpn_cert',
        arguments: { user: 'jdoe', reason: 'ticket #8821' },
        result: 'cert reissued · serial 0x4F…A1 · expires 2027-06-03',
        error: null,
      },
    ],
    has_versions: false,
  },
  {
    interaction_id: 1045,
    role: 'user',
    content: 'Great. Email Jane the confirmation too.',
    ephemeral: false,
    reasoning: null,
    tool_context: null,
    has_versions: false,
  },
  {
    interaction_id: null,
    ephemeral_chunk_id: 'pend_7f3c',
    role: 'assistant',
    content:
      'About to run email_user(to="jdoe@corp", template="cert_confirm"). This is a write — approve to send.',
    ephemeral: true,
    reasoning: null,
    tool_context: [
      {
        call_id: 'c_b4e0',
        group_id: 'g_2',
        tool_name: 'email_user',
        arguments: { to: 'jdoe@corp', template: 'cert_confirm', ticket: '8821' },
        result: null,
        error: null,
      },
    ],
    has_versions: false,
  },
]

export const MOCK_VERSIONS_1042: VersionsResponse = {
  interaction_id: 1042,
  versions: [
    { version: 1, content: 'Resetting now.', canonical: false },
    {
      version: 2,
      content: 'I can reset it, but DP-204 wants L2 sign-off first. Proceed?',
      canonical: false,
    },
    {
      version: 3,
      content:
        "Heads up — reissuing a VPN cert is gated by policy DP-204 (L2 approval). I've staged the reset; approve below and I'll run it.",
      canonical: true,
    },
  ],
}

// Offline fallback for the Raw-req inspector. The source is deliberately
// 'client_fallback' so the parity banner goes RED when the engine is
// unreachable — the mock is NOT the shared live builder and could drift.
export const MOCK_ASSEMBLED: AssembledRequest = {
  parity: {
    source: 'client_fallback',
    builder: 'mock',
    matches_live: false,
  },
  route: 'engine · POST /v1/chat/completions',
  model_name: MOCK_PERSONA.model_name,
  params: {
    temperature: MOCK_PERSONA.temperature,
    top_p: MOCK_PERSONA.top_p,
    top_k: MOCK_PERSONA.top_k,
    max_tokens: MOCK_PERSONA.max_tokens,
    stop: null,
    seed: null,
    rep_pen: MOCK_PERSONA.kobold_extras.rep_pen,
    min_p: MOCK_PERSONA.kobold_extras.min_p,
    tfs: MOCK_PERSONA.kobold_extras.tfs,
  },
  messages: [
    { role: 'system', content: MOCK_PERSONA.prompt, src: 'persona.prompt' },
    {
      role: 'user',
      content: MOCK_LTM_BLOCK ?? '<memory>…</memory>',
      src: 'ltm_block',
    },
    { role: 'user', content: 'Reset the VPN cert for jdoe.', src: '#1041' },
    {
      role: 'assistant',
      content: 'Reissuing a VPN cert is gated by policy DP-204 (L2 approval).',
      src: '#1042 · v3 canonical',
    },
    { role: 'user', content: 'Email Jane the confirmation too.', src: 'composer' },
  ],
}
