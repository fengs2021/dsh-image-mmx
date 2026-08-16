/**
 * dsh-image-mmx
 *
 * 让文本模型（如 DeepSeek，inputModalities 只有 text）也能正常粘贴并发送图片：
 *  - 图片落盘到会话工作区 `.attachments/`；
 *  - 图片块保留在消息里（客户端照常渲染缩略图、可点击放大）；
 *  - 通过 DSH surface replace 机制，把图片在「模型可见面」替换成
 *    `[图片N]:"<路径>"` + mmx（MiniMax VLM）自动识别结果文本，
 *    模型无需视觉能力即可理解图片内容；
 *  - mmx 识别失败时降级为路径文本（模型仍可自行调用其他视觉工具）；
 *  - 支持图片的模型不做任何处理，走原生路径。
 *
 * 依赖：本机已安装并认证 mmx-cli（~/.mmx/config.json 有 api_key）。
 */

import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

const name = "image-mmx";

const inject = ["attachments", "llm", "sandboxPolicy"];

const B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

function bytesToBase64(bytes) {
  let out = "";
  for (let i = 0; i < bytes.length; i += 3) {
    const b0 = bytes[i];
    const b1 = i + 1 < bytes.length ? bytes[i + 1] : -1;
    const b2 = i + 2 < bytes.length ? bytes[i + 2] : -1;
    out += B64[b0 >> 2];
    out += B64[((b0 & 3) << 4) | (b1 === -1 ? 0 : b1 >> 4)];
    out += b1 === -1 ? "=" : B64[((b1 & 15) << 2) | (b2 === -1 ? 0 : b2 >> 6)];
    out += b2 === -1 ? "=" : B64[b2 & 63];
  }
  return out;
}

function extFor(mediaType) {
  if (mediaType === "image/jpeg") return ".jpg";
  if (mediaType === "image/webp") return ".webp";
  if (mediaType === "image/gif") return ".gif";
  return ".png";
}

function shaOf(ref) {
  const id = String(ref && ref.attachmentId ? ref.attachmentId : "");
  return id.indexOf("sha256:") === 0 ? id.slice(7) : id;
}

function toPosix(p) {
  return String(p).replace(/\\/g, "/");
}

/** mmx 不支持的格式（gif 等）直接跳过识别，降级为路径文本。 */
const MMX_SUPPORTED = new Set(["image/jpeg", "image/png", "image/webp"]);

/** 单张图片识别超时（毫秒）。 */
const MMX_TIMEOUT_MS = 60_000;

/** 识别提示词：要求详细中文描述，保留文字信息（报错/代码/界面文本）。 */
const MMX_PROMPT =
  "请用中文详细描述这张图片的内容，包括画面元素、布局、以及图片中的全部文字信息（如报错信息、代码、界面文案）。如果图片里有代码或错误信息，请逐字保留。";

/** mmx 可执行文件候选（systemd 环境 PATH 可能不含 /usr/local/bin）。 */
const MMX_BIN_CANDIDATES = ["mmx", "/usr/local/bin/mmx"];

/**
 * 调用本机 mmx CLI 识别图片。
 * @param {string} imagePath 本地图片路径（jpg/png/webp）
 * @returns {Promise<{ok:boolean, text:string}>}
 */
function runMmxVision(imagePath) {
  return new Promise((resolve) => {
    let binIndex = 0;
    let child = null;
    let stdout = "";
    let stderr = "";
    let settled = false;

    const finish = (ok, text) => {
      if (settled) return;
      settled = true;
      resolve({ ok, text });
    };

    const timer = setTimeout(() => {
      try {
        child && child.kill("SIGKILL");
      } catch (_) {
        /* ignore */
      }
      finish(false, "");
    }, MMX_TIMEOUT_MS);

    const wire = () => {
      child.stdout.on("data", (d) => {
        stdout += d;
      });
      child.stderr.on("data", (d) => {
        stderr += d;
      });
      child.on("error", () => {
        // 第一个候选找不到时尝试绝对路径
        if (binIndex < MMX_BIN_CANDIDATES.length - 1) {
          binIndex += 1;
          try {
            child = spawn(
              MMX_BIN_CANDIDATES[binIndex],
              ["vision", "describe", "--image", imagePath, "--prompt", MMX_PROMPT, "--quiet"],
              { stdio: ["ignore", "pipe", "pipe"] }
            );
            stdout = "";
            stderr = "";
            wire();
          } catch (err) {
            finish(false, "");
          }
        } else {
          finish(false, "");
        }
      });
      child.on("close", (code) => {
        clearTimeout(timer);
        let text = stdout.trim();
        if (code === 0 && text.length > 0) {
          // mmx --quiet 输出 JSON（content + base_resp），提取 content 纯文本
          if (text.startsWith("{")) {
            try {
              const parsed = JSON.parse(text);
              if (typeof parsed.content === "string" && parsed.content.length > 0) {
                text = parsed.content;
              }
            } catch (_) {
              /* 非 JSON 则原样使用 */
            }
          }
          finish(true, text);
        } else {
          if (stderr) {
            console.warn(
              `image-mmx: mmx vision failed (exit ${code}): ${stderr.slice(0, 300)}`
            );
          }
          finish(false, "");
        }
      });
    };

    try {
      child = spawn(
        MMX_BIN_CANDIDATES[binIndex],
        ["vision", "describe", "--image", imagePath, "--prompt", MMX_PROMPT, "--quiet"],
        { stdio: ["ignore", "pipe", "pipe"] }
      );
      wire();
    } catch (err) {
      finish(false, "");
    }
  });
}

function apply(ctx) {
  const attachments = ctx.attachments;
  const llm = ctx.llm;
  const sandboxPolicy = ctx.sandboxPolicy;

  // Part 1: 让 api-proxy 的预检放行含图 prompt（对文本模型）。
  // 预检用 `llm.resolveModelInfo(...).inputModalities` 判断是否支持图片；
  // 这里把文本模型的结果补上 "image"，从而让消息能进入 agent 流程。
  let originalResolve;
  ctx.effect(() => {
    if (typeof llm.resolveModelInfo !== "function") return () => {};
    originalResolve = llm.resolveModelInfo.bind(llm);
    llm.resolveModelInfo = async (provider, model, signal) => {
      const info = await originalResolve(provider, model, signal);
      if (
        info &&
        Array.isArray(info.inputModalities) &&
        !info.inputModalities.includes("image")
      ) {
        return { ...info, inputModalities: [...info.inputModalities, "image"] };
      }
      return info;
    };
    return () => {
      try {
        delete llm.resolveModelInfo;
      } catch (_) {
        /* 还原失败忽略：实例会随进程退出 */
      }
    };
  });

  function workspaceRootFor(agent) {
    try {
      const policy = sandboxPolicy.resolve({ session: agent && agent.session });
      if (
        policy &&
        typeof policy.workspaceRoot === "string" &&
        policy.workspaceRoot.length > 0
      ) {
        return policy.workspaceRoot;
      }
    } catch (_) {
      /* fall through */
    }
    if (typeof sandboxPolicy.workspaceRoot === "string") return sandboxPolicy.workspaceRoot;
    return "";
  }

  function currentModel(agent) {
    try {
      const header = agent && agent.session ? agent.session.requestHeader() : undefined;
      const cfg = header && header.config;
      if (cfg && cfg.provider && cfg.model) return { provider: cfg.provider, model: cfg.model };
    } catch (_) {
      /* fall through */
    }
    if (agent && agent.options && agent.options.provider && agent.options.model) {
      return { provider: agent.options.provider, model: agent.options.model };
    }
    return null;
  }

  async function writeImageFile(data, destPath) {
    try {
      mkdirSync(dirname(destPath), { recursive: true });
      writeFileSync(destPath, Buffer.from(bytesToBase64(data), "base64"));
      return true;
    } catch (err) {
      console.warn(
        "image-mmx: write failed:",
        String(err && err.message ? err.message : err)
      );
      return false;
    }
  }

  // 每个 step 的状态：attachmentId -> { path, description? }
  // 内容寻址，按 step 重置。
  let imageMeta = new Map();

  async function modelSupportsImages(agent) {
    if (typeof originalResolve !== "function") return false;
    const cur = currentModel(agent);
    if (!cur) return false;
    try {
      const info = await originalResolve(cur.provider, cur.model);
      return !!(
        info &&
        Array.isArray(info.inputModalities) &&
        info.inputModalities.includes("image")
      );
    } catch (_) {
      return false;
    }
  }

  // Part 2: 写盘 + mmx 自动识别，但保留图片块（客户端渲染缩略图）。
  ctx.on("agent/pre-step", async (payload, next) => {
    const decision = await next();
    if (!decision || decision.kind !== "enter" || !Array.isArray(decision.messages)) {
      return decision;
    }
    imageMeta = new Map();
    if (!(await modelSupportsImages(payload.agent))) {
      const root = workspaceRootFor(payload.agent);
      const tasks = [];
      for (const message of decision.messages) {
        if (!message || message.role !== "user" || !Array.isArray(message.content)) continue;
        for (const block of message.content) {
          if (!block || block.type !== "image") continue;
          const id = String(
            block.attachment && block.attachment.attachmentId
              ? block.attachment.attachmentId
              : ""
          );
          tasks.push(
            (async () => {
              try {
                const stored = await attachments.readImage(block.attachment, payload.signal);
                const ref = stored.ref;
                const base =
                  shaOf(ref) || "img-" + Date.now() + "-" + Math.random().toString(16).slice(2);
                const destPath =
                  toPosix(root).replace(/\/+$/, "") + "/.attachments/" + base + extFor(ref.mediaType);
                const ok = await writeImageFile(stored.data, destPath);
                const meta = { path: destPath, description: "" };
                if (ok && id) {
                  imageMeta.set(id, meta);
                  // mmx 自动识别（并行，失败不阻断落盘流程）
                  if (MMX_SUPPORTED.has(ref.mediaType)) {
                    const result = await runMmxVision(destPath);
                    if (result.ok) meta.description = result.text;
                  }
                }
              } catch (err) {
                console.warn(
                  "image-mmx: prepare failed:",
                  String(err && err.message ? err.message : err)
                );
              }
            })()
          );
        }
      }
      await Promise.allSettled(tasks);
    }
    return decision;
  });

  function buildTextVersion(message) {
    const newContent = [];
    let idx = 0;
    for (const block of message.content) {
      if (block && block.type === "image") {
        idx += 1;
        const id = String(
          block.attachment && block.attachment.attachmentId
            ? block.attachment.attachmentId
            : ""
        );
        const meta = imageMeta.get(id);
        if (meta && meta.path) {
          let text = '[图片' + idx + ']:"' + meta.path + '"';
          if (meta.description) {
            text += "\n[mmx 视觉识别] " + meta.description;
          }
          newContent.push({ type: "text", text });
        } else {
          newContent.push({
            type: "text",
            text: '[图片' + idx + ']:"(保存失败)"',
          });
        }
      } else {
        newContent.push(block);
      }
    }
    return { ...message, content: newContent };
  }

  // 从日志尾部向前扫描当前 step 内的图片消息（停在 step/start）。
  function findImageMessages(agent) {
    const events = agent && agent.session ? agent.session.events : [];
    const targets = [];
    for (let i = events.length - 1; i >= 0; i--) {
      const ev = events[i];
      if (ev.type === "step/start") break;
      if (ev.type !== "user/message") continue;
      const msg = ev.data;
      if (!msg || !Array.isArray(msg.content)) continue;
      for (const block of msg.content) {
        if (block && block.type === "image") {
          const id = String(
            block.attachment && block.attachment.attachmentId
              ? block.attachment.attachmentId
              : ""
          );
          if (imageMeta.has(id)) {
            targets.push(ev);
            break;
          }
        }
      }
    }
    targets.reverse();
    return targets;
  }

  // Part 3: 只要当前 step 还有「已落盘但未替换」的图片消息，就追加 model-only replace 并 retry。
  ctx.on("agent/request-error", async (payload, next) => {
    if (imageMeta.size === 0) return next();
    const targets = findImageMessages(payload.agent);
    if (targets.length === 0) {
      imageMeta = new Map();
      return next();
    }
    for (const t of targets) {
      const replacement = buildTextVersion(t.data);
      try {
        payload.agent.session.append("user/message", replacement, {
          surfaceOp: { op: "replace", start: t.seq, end: t.seq },
          sourceEventSeqs: [t.seq],
        });
      } catch (err) {
        console.warn(
          "image-mmx: replace failed:",
          String(err && err.message ? err.message : err)
        );
      }
    }
    imageMeta = new Map();
    return { kind: "retry" };
  });
}

export { apply, inject, name };
