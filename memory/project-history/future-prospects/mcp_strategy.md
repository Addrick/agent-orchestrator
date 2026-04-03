---
name: MCP integration strategy
description: Phased path from MCP client to engine-as-server — with LSP patterns as architectural inspiration
type: project
---

**Created:** 2026-03-24

## Why MCP
Solves M*N integration problem: any agent talks to any tool server via standardized JSON-RPC. Ecosystem of existing servers (GitHub, Slack, Linear, Jira, Notion, etc.).

## Integration Path
- **Phase 1:** MCP Client — consume existing MCP servers, expose their tools to LLM
- **Phase 2:** Wrap Zammad as MCP Server — extract ZammadToolHandler into standalone server
- **Phase 3:** Engine as MCP Server — expose engine capabilities for other agents
- **Phase 4:** MCP Server Discovery / Registry — MSPs add servers via config

## Component Evolution
- `ServiceIntegration` ABC -> MCP server interface
- `ToolManager` + `definitions.py` -> Routes to both native tools and MCP servers
- `ZammadIntegration` -> First candidate for standalone MCP server
- `TextEngine` -> Stays as-is (LLM layer, not a tool)

## Key Decisions
1. Start Phase 1 after tool permission gating is built
2. Python MCP SDK (`mcp` package, Anthropic-maintained) is adequate for client use
3. Don't over-abstract: ServiceIntegration ABC stays for engine-internal integrations, MCP is additive
