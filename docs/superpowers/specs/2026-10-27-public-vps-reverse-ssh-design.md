# 公网 VPS 反向 SSH：外网访问 wyf（MiniSeek + SSH）

Date: 2026-10-27  
Status: Approved（grill 对齐：方案 A 反向 SSH；跳转端口 2222；Web 走 VPS :80→wyf :7860；autossh + systemd user；密钥登录）  
Repo: `wyf-yfw/miniseek`  
Machine: wyf（隧道客户端）；VPS `60.205.250.186`（公网入口）  
Depends on: 三服务 Web（frontend `:7860` / api `:8000` / worker `:8001`）；本设计只改对外入口，不改三服务边界

## Goal

让用户在公网用 **浏览器** 打开 MiniSeek，并用 **SSH** 登录家里的 3090 机器（wyf），**不依赖** Tailscale Funnel。

固定映射：

| 公网（VPS） | 落到 | 用途 |
|-------------|------|------|
| `60.205.250.186:80` | wyf `:7860` | MiniSeek 前端（静态 + `/api` 反代） |
| `60.205.250.186:2222` | wyf `:22` | SSH 跳到内网 3090 |
| `60.205.250.186:22` | VPS 本机 sshd | 管理公网机本身 |

实现手段：**方案 A**——wyf 主动向 VPS 建立 **反向 SSH 隧道**，用 `autossh` + systemd user unit 常驻、断线重连；密钥认证；VPS 开启 `GatewayPorts` / `AllowTcpForwarding`。

## Non-goals

- 本轮 **只写设计**；不落地 unit、不改 sshd、不装软件、不推送密钥。
- 不引入 FRP / nps / WireGuard / 商业内网穿透；若日后替换，另开 spec。
- 不把 VPS 变成第二套 MiniSeek 运行环境；VPS **不跑** api/worker/GPU，只做入口与转发。
- 不在本设计里开 HTTPS / 申请证书（可后续加；首版明文 `:80` 可接受，风险见下）。
- 不暴露 wyf 的 `:8000` / `:8001` 到公网；外网只碰前端 `:7860` 与 SSH。
- 不强制拆除 Funnel；公网入口就绪后 Funnel **可选**，对「公网访问」不再必需。
- **文档与仓库不写任何明文密码**；VPS 登录凭据已在保险库，实现时从 vault 取用。

## Context

- wyf 在内网 / NAT 后，公网无法直连其 `:22` / `:7860`。
- MiniSeek 本机已拆成三服务；对外入口仍是 wyf `:7860`（见 `2026-10-27-three-service-web-design.md`）。
- 用户另有公网 VPS `60.205.250.186`，希望：浏览器打公网 IP 用 Web；`ssh -p 2222` 进 3090。
- 反向 SSH 由 **内网主动连出**，适合家宽 NAT，无需在路由器做端口映射。

## Architecture

```text
  Internet
      │
      ├─ browser ──► VPS:80  ──(ssh -R)──► wyf:7860  (miniseek-web)
      │                                      │
      │                                      ├─ /api → 127.0.0.1:8000 (api)
      │                                      └─ api → 127.0.0.1:8001 (worker)
      │
      ├─ ssh -p 2222 user@VPS ──(ssh -R)──► wyf:22
      │
      └─ ssh -p 22 admin@VPS  ────────────► VPS 本机管理

  wyf (systemd --user)
      miniseek-tunnel.service
         autossh → VPS:22 (key)
            RemoteForward 0.0.0.0:80   → 127.0.0.1:7860
            RemoteForward 0.0.0.0:2222 → 127.0.0.1:22
```

### 角色

| 角色 | 机器 | 职责 |
|------|------|------|
| **隧道客户端** | wyf | `autossh` 连 VPS；携带两条 `-R`；systemd user 保活 |
| **隧道服务端 / 入口** | VPS | sshd 接受反向绑定；防火墙放行 22/80/2222；可选本机 nginx 仅当日后要 TLS 时再加（首版 **直接** 由 `GatewayPorts` 把 `:80`/`:2222` 绑到公网网卡即可） |
| **业务** | wyf | 既有 `miniseek-web` / `miniseek-api` / `miniseek-worker`；隧道不替代它们 |

### 为何不用「公网机主动连内网」

家宽 / NAT 下 VPS 通常 **不能** 直连 wyf；反向隧道由 wyf 出站到 VPS:22，天然穿透。FRP 等同思路但多组件；本设计锁死 SSH 反连以降低运维面。

## Data / control flow

1. wyf 启动 `miniseek-tunnel` → `autossh` 用专用密钥登录 VPS → 在 VPS 上监听 `0.0.0.0:80` 与 `0.0.0.0:2222`，分别转发到 wyf 本机 `127.0.0.1:7860` 与 `127.0.0.1:22`。
2. 用户浏览器访问 `http://60.205.250.186/` → 流量经隧道到 `miniseek-web` → `/api` 仍只打本机 api；训练进程仍只归 worker。
3. 用户 `ssh -p 2222 <wyf-user>@60.205.250.186` → 落到 wyf 的 sshd（认证是 **wyf 的用户/密钥**，不是 VPS root 会话里的 shell 转发身份混淆：远程端口只是 TCP 接到 wyf:22）。
4. 管理员 `ssh -p 22 <vps-admin>@60.205.250.186` → 只管理 VPS。
5. 隧道断开：`autossh` 重连；重连期间 Web/SSH 跳转短暂不可用，**不影响** 已在 wyf 上跑的训练（与三服务设计一致）。
6. Funnel：可继续指 `:7860` 作备用；公网主入口改为本 VPS 映射后，文档与运维以 VPS 为准。

## VPS 配置要求（实现时落地，本设计给定目标态）

sshd（示意，实现时改 `/etc/ssh/sshd_config` 或 drop-in）：

- `AllowTcpForwarding yes`
- `GatewayPorts yes`（或 `clientspecified`，且客户端 `-R` 使用 `0.0.0.0:port:...`）
- 密钥登录可用后：`PasswordAuthentication no`（或至少禁止 root 密码）；保留密钥与（如需要）控制台救援路径
- 监听 `:22` 供管理与隧道入站；**不要**在 VPS 上再开第二套业务 sshd 抢 2222——2222 由反向隧道占用

防火墙 / 安全组（公网只放行）：

| 端口 | 协议 | 谁连 | 说明 |
|------|------|------|------|
| 22 | TCP | 管理员 | VPS 本机 SSH |
| 80 | TCP | 浏览器 | → wyf MiniSeek |
| 2222 | TCP | 用户 SSH 客户端 | → wyf SSH |
| 其余 | — | 拒绝 | 尤其勿对公网开 8000/8001 |

账号与密钥：

- VPS 上为隧道建 **专用用户**（例如 `tunnel`），仅允许密钥、可限制 `ForceCommand`/`permitopen` 到转发所需（实现阶段收紧；最低要求：独立密钥、无 sudo 也可工作若只做 `-R`）。
- wyf 上生成 ed25519（或现有）密钥，**私钥只留 wyf**；公钥装到 VPS 隧道用户 `authorized_keys`。
- 仓库与本 spec **不提交私钥、不写密码**。

注意：Linux 上绑定远程 `:80` 通常需要 VPS 上 sshd 以足够权限允许非特权用户转发到特权端口，或改用 `setcap`/authbind/先绑高位再本机 iptables 转发。实现时若 `:80` 绑定失败，允许等价方案：**VPS 本机 nginx/Caddy 监听 `:80`，proxy 到 `127.0.0.1:<高位>`**，而 autossh 只 `-R 127.0.0.1:<高位>:127.0.0.1:7860`。对外契约仍是「公网 `:80` → MiniSeek」；对内多一层本机反代不算改 Goal。`:2222` 非特权，可直接 `-R 0.0.0.0:2222:127.0.0.1:22`。

## wyf 配置要求

- 安装 `autossh`（发行版包即可）。
- systemd **user** unit（建议名 `miniseek-tunnel.service`），示意行为：
  - `ExecStart`：`autossh -M 0 -N` + ServerAlive 选项 + 两条 `RemoteForward` + `tunnel@60.205.250.186`（用户名以实现时为准）
  - `Restart=always`；网络差时依赖 autossh 重连
  - 不依赖 `miniseek-web` 的生命周期：web 重启不断隧道；隧道重启不杀 GPU
- wyf 本机 sshd 保持 `:22` 仅监听合理地址（可 `0.0.0.0` 或内网）；公网不直连它，只经 VPS:2222。
- MiniSeek 三服务保持现状；隧道只 consumewy `:7860` 与 `:22`。

### 建议的 SSH 客户端选项（实现写入 unit / `~/.ssh/config`）

- `ServerAliveInterval 30`、`ServerAliveCountMax 3`
- `ExitOnForwardFailure yes`（转发失败则退出，交给 systemd/autossh 重来，避免「连着但没端口」的假活）
- `IdentitiesOnly yes` + 指定 IdentityFile

## 用户侧用法（验收口径）

- Web：`http://60.205.250.186/`（或日后 `https://…`）应打开与本机 `:7860` 相同的 MiniSeek UI；能登录后的任务流与三服务验收一致。
- SSH：`ssh -p 2222 <wyf-user>@60.205.250.186` 进入 wyf shell（密钥或 wyf 本机认证策略）。
- VPS 管理：`ssh -p 22 <vps-admin>@60.205.250.186`。

## Funnel 关系

| 入口 | 状态 |
|------|------|
| Tailscale Funnel → wyf `:7860` | **可选备用**；公网主路径改为 VPS 后可不依赖 |
| VPS `:80` → wyf `:7860` | **公网主入口**（本设计） |
| 直连内网 IP | 仅局域网；不替代本设计 |

## Security

- 公网面：22 / 80 / 2222；关闭无用端口。
- 密钥优于密码；密码登录在密钥验证通过后关闭；**任何密码不入库、不进 chat 副本、不进本文件**。
- Web 首版 HTTP：存在窃听与篡改风险；实现后优先排期 TLS（Let’s Encrypt + VPS 反代）作为 follow-up，不阻塞首版打通。
- 暴露的是完整 MiniSeek UI（含启停训练能力）：等同把控制面放到公网。缓解（实现可选、本设计不强制一次做完）：HTTP Basic / 反代鉴权、IP allowlist、或仅 VPN 后再关公网 80——若首版必须公网裸开，须在实现 PR 注明「已知风险，用户知情」。
- 隧道用户权限最小化；定期轮换密钥。
- 聊天中曾出现过的 VPS 密码视为已泄露线索：实现前后在控制台/保险库 **改密**，本 spec 不记录旧密。

## Acceptance

1. 从外网（非 wyf 局域网）浏览器打开 `http://60.205.250.186/`，MiniSeek UI 可用；`/api` 健康或任务列表可读。
2. 外网 `ssh -p 2222 <wyf-user>@60.205.250.186` 登录到 **wyf**（`hostname` / 家目录可证），不是停在 VPS。
3. 外网 `ssh -p 22 <vps-admin>@60.205.250.186` 仍只管理 VPS。
4. `systemctl --user restart miniseek-web`（或 api）后，隧道仍在；训练若在跑则不被隧道或 web 重启杀死（回归三服务验收）。
5. 人为断开隧道进程后，unit/autossh 自动恢复，80 与 2222 再次可用。
6. VPS 安全组/防火墙未对公网开放 8000/8001；`ss`/`firewall` 抽查通过。
7. 密钥登录 VPS 隧道用户成功后，密码登录按计划关闭（或已关闭）。
8. 仓库与本文件无明文密码、无私钥。

## Risks

- `:80` 特权端口绑定失败 → 用「VPS 本机反代 → 高位 localhost 转发」等价实现，对外仍是 `:80`。
- 家宽上行带宽不足 → Web SSE/大日志卡顿；不改架构，可压缩日志或限流（另议）。
- 运营商干扰 80 → 可改对外 8080 或上 TLS 443（follow-up）；本设计默认 80。
- 双入口（Funnel + VPS）同时开 → 行为一致但运维易混；文档标明主入口为 VPS。
- `GatewayPorts yes` 全局偏宽 → 实现时优先收紧到隧道用户 / `permitopen`。
- 公网裸 HTTP 控制面 → 见 Security；不阻塞打通，但必须在实现说明里写明。

## Success criteria（总结）

- 公网 IP 浏览器可达 MiniSeek；`ssh -p 2222` 进 wyf；`ssh -p 22` 管 VPS。
- 机制为 wyf→VPS 反向 SSH + autossh systemd user；密钥认证；GatewayPorts/转发已开。
- 不依赖 Funnel 做公网访问；三服务与训练生命周期不变。
- 机密只在保险库，不在仓库。

## Implementation notes（非本提交）

实现阶段建议顺序：VPS sshd/防火墙与隧道用户 → wyf 密钥与手动 `ssh -R` 冒烟 → autossh user unit → 验收表 →（可选）关密码登录与 Funnel 降级说明。单元文件与运维命令可放 `deploy/` 或文档；**本 commit 仅本设计文件**。
