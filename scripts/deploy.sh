#!/usr/bin/env bash
# MiniSeek 部署脚本 —— 在 GPU 机（wyf）上执行
#
# 用法：
#   ./scripts/deploy.sh                 拉代码 + 按训练状态重启服务
#   ./scripts/deploy.sh --dry-run       只打印将要做什么，不动手
#   ./scripts/deploy.sh --restart       没有新提交也强制重启
#   ./scripts/deploy.sh --skip-pull     不拉代码，只重启
#   ./scripts/deploy.sh --force-worker  即使训练中也重启 worker（会杀训练！）
#
# 返回码：0 成功 / 1 前置检查失败 / 2 部署后健康检查失败
set -uo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BRANCH="${BRANCH:-main}"
API_URL="${API_URL:-http://127.0.0.1:8000}"
WEB_URL="${WEB_URL:-http://127.0.0.1:7860}"
WORKER_URL="${WORKER_URL:-http://127.0.0.1:8001}"

DRY_RUN=0
FORCE_WORKER=0
SKIP_PULL=0
FORCE_RESTART=0

log()  { printf '\033[36m[deploy]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[deploy]\033[0m %s\n' "$*"; }
err()  { printf '\033[31m[deploy]\033[0m %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }

run() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '  \033[90m(dry-run) %s\033[0m\n' "$*"
  else
    "$@"
  fi
}

for arg in "$@"; do
  case "$arg" in
    --dry-run)      DRY_RUN=1 ;;
    --force-worker) FORCE_WORKER=1 ;;
    --skip-pull)    SKIP_PULL=1 ;;
    --restart)      FORCE_RESTART=1 ;;
    -h|--help)      sed -n '2,11p' "$0"; exit 0 ;;
    *)              die "未知参数：$arg（用 --help 看用法）" ;;
  esac
done

# ---------------------------------------------------------------- 1. 前置检查
cd "$REPO_DIR" 2>/dev/null || die "仓库目录不存在：$REPO_DIR"
[ -d .git ] || die "$REPO_DIR 不是 git 仓库"

command -v git >/dev/null || die "找不到 git"
command -v curl >/dev/null || die "找不到 curl"

if [ "$SKIP_PULL" = "0" ]; then
  if [ -n "$(git status --porcelain)" ]; then
    warn "工作区有未提交改动，为了不覆盖它们，先停下来："
    git status --short
    die "请先 commit / stash，或加 --skip-pull 跳过拉取"
  fi
fi

OLD_REV="$(git rev-parse --short HEAD)"
log "当前版本：$OLD_REV （分支 $(git rev-parse --abbrev-ref HEAD)）"

# ---------------------------------------------------------------- 2. 拉代码
CHANGED=1
if [ "$SKIP_PULL" = "1" ]; then
  log "按参数跳过 git pull"
else
  log "拉取 origin/$BRANCH ..."
  run git fetch --prune origin || die "git fetch 失败（网络或凭据问题）"
  if ! run git merge --ff-only "origin/$BRANCH"; then
    die "无法快进合并。远端可能分叉，请手动处理后再部署"
  fi
fi

NEW_REV="$(git rev-parse --short HEAD)"
if [ "$OLD_REV" = "$NEW_REV" ]; then
  CHANGED=0
  log "代码无变化，仍是 $NEW_REV"
else
  log "代码更新：$OLD_REV -> $NEW_REV"
fi

if [ "$CHANGED" = "0" ] && [ "$FORCE_RESTART" = "0" ] && [ "$SKIP_PULL" = "0" ]; then
  log "没有新提交，不重启服务。要强制重启请加 --restart"
  exit 0
fi

# ---------------------------------------------------------------- 3. 训练状态
# 注意：这里既用来决定重启范围，也是唯一能防止误杀训练的检查。
probe_training() {
  local json
  json="$(curl -fsS -m 5 "$API_URL/api/status" 2>/dev/null)" || return 2
  printf '%s' "$json" | grep -q '"running"[[:space:]]*:[[:space:]]*true'
}

TRAINING=0
probe_training
PROBE_RC=$?
case "$PROBE_RC" in
  0) TRAINING=1 ;;
  2) warn "API 无响应（$API_URL/api/status），无法判断训练状态" ;;
esac

if [ "$TRAINING" = "1" ]; then
  log "检测到训练正在运行，worker 将保持不变"
else
  log "当前没有训练在跑"
fi

# ---------------------------------------------------------------- 4. 重启服务
# 铁律：不要重启 miniseek-tunnel。它承载着 7860 转发和 2222 这条 SSH，
# 断开它会把正在执行本脚本的连接一起切断。
UNITS_WEB_API="miniseek-api miniseek-web"

restart_units() {
  log "重启：$*"
  if [ "$DRY_RUN" = "0" ]; then
    if ! systemctl --user restart "$@"; then
      err "systemctl --user 失败。若提示连接不上 bus，需要在带 XDG_RUNTIME_DIR 的环境里跑："
      err "  export XDG_RUNTIME_DIR=/run/user/$(id -u)"
      exit 1
    fi
  else
    printf '  \033[90m(dry-run) systemctl --user restart %s\033[0m\n' "$*"
  fi
}

if [ "$TRAINING" = "1" ] && [ "$FORCE_WORKER" = "0" ]; then
  restart_units $UNITS_WEB_API
  warn "worker 未重启 —— 训练进程归它持有，重启会直接杀掉训练"
  warn "训练结束后想更新 worker，跑：./scripts/deploy.sh --skip-pull --restart"
elif [ "$TRAINING" = "1" ] && [ "$FORCE_WORKER" = "1" ]; then
  warn "==== 已指定 --force-worker：训练会被中断 ===="
  restart_units $UNITS_WEB_API miniseek-worker
else
  restart_units $UNITS_WEB_API miniseek-worker
fi

# ---------------------------------------------------------------- 5. 健康检查
if [ "$DRY_RUN" = "1" ]; then
  log "dry-run 结束，没有改动任何服务"
  exit 0
fi

wait_http() {
  local url="$1" name="$2" tries="$3" i
  for ((i = 1; i <= tries; i++)); do
    if curl -fsS -m 3 -o /dev/null "$url" 2>/dev/null; then
      printf '  %-8s OK   %s\n' "$name" "$url"
      return 0
    fi
    sleep 1
  done
  printf '  %-8s FAIL %s\n' "$name" "$url"
  return 1
}

log "健康检查 ..."
FAILED=0
wait_http "$WORKER_URL/health" worker 20 || FAILED=1
wait_http "$API_URL/api/status" api 20 || FAILED=1
wait_http "$WEB_URL/" web 20 || FAILED=1

if [ "$FAILED" = "1" ]; then
  err "有服务没起来。回滚命令："
  err "  cd $REPO_DIR && git reset --hard $OLD_REV"
  err "  systemctl --user restart miniseek-api miniseek-web"
  err "  journalctl --user -u miniseek-api -u miniseek-web -n 50 --no-pager"
  exit 2
fi

# ---------------------------------------------------------------- 6. 汇总
log "部署完成：$OLD_REV -> $NEW_REV"
[ "$TRAINING" = "1" ] && [ "$FORCE_WORKER" = "0" ] && warn "worker 仍在跑旧代码，训练结束后再重启它"
exit 0
