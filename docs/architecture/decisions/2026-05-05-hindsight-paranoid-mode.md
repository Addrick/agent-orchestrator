---
name: Hindsight paranoid-mode security posture (DP-109)
description: Why DP-109 used a hand-audited httpx client + air-gapped Docker network + SHA-pinned image + cap_drop ALL + named volume — supply-chain-incident-era threat model
type: project
---

# Security Reasoning: Hindsight Backend (DP-109)

## Threat Model
The Hindsight integration was performed during a period of elevated supply chain risk (Ref: TeamPCP attacks on `liteLLM`, `axios`, and `npm`). 

### Primary Threats
1.  **Malicious Client Installer:** 3rd-party Python packages often use `setup.py` or `.pth` files to execute code during installation. A compromised `hindsight-client` could infect the host environment before any application code runs.
2.  **Data Exfiltration (Phone Home):** A compromised server or client could attempt to exfiltrate project data or memory snippets to an external command-and-control server.
3.  **Container Escape:** A vulnerability in the Hindsight image could be exploited to attempt a breakout from the Docker container to the Windows host.

## Mitigation Strategy: "Paranoid Mode"

### 1. Zero-Trust Client (Aped SDK)
- **Action:** We chose **not** to install the `hindsight-client` package.
- **Reasoning:** By manually auditing and implementing the Hindsight REST logic using the project's existing and trusted `httpx` library, we eliminate the installation-time attack surface. We gain full control over the bytes sent to the server.

### 2. Air-Gapped Infrastructure
- **Action:** Docker containers are configured with `internal: true` networks.
- **Reasoning:** By denying internet egress, we prevent the containerized service from communicating with the outside world. This neutralizes any hidden exfiltration logic or "phone home" features.

### 3. Immutable Verification
- **Action:** Every image is pinned using its unique SHA256 digest (`sha256:a435...`).
- **Reasoning:** Tagging (e.g., `:latest`) is vulnerable to tag-poisoning. Pinning by SHA ensures that the exact audited image is deployed and cannot be silently replaced by a malicious update.

### 4. Privilege Minimization
- **Action:** Container runs with `cap_drop: ALL` and `no-new-privileges: true`.
- **Reasoning:** Even if an attacker gains execution within the container, they lack the Linux capabilities (like `CAP_SYS_ADMIN` or `CAP_NET_RAW`) required to perform lateral movement or exploit host-level kernel vulnerabilities.

### 5. Filesystem Isolation
- **Action:** Used named Docker volumes (`hindsight-data`) instead of host bind mounts (`C:\...`).
- **Reasoning:** Bind mounts provide a direct bridge to the host filesystem. Named volumes ensure that the container only sees its own virtual disk, preventing it from ever "seeing" the Pycharm project directory or other host sensitive files.
