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
  console.log("[image-mmx] apply() 执行: attachments?", !!ctx.attachments, "llm.resolveModelInfo?", typeof (ctx.llm && ctx.llm.resolveModelInfo));

  // Part 1: 让 api-proxy 的预检放行含图 prompt（对文本模型）。
  // 预检用 `llm.resolveModelInfo(...).inputModalities` 判断是否支持图片；
  // 这里把文本模型的结果补上 "image"，从而让消息能进入 agent 流程。
  let originalResolve;
  ctx.effect(() => {
    if (typeof llm.resolveModelInfo !== "function") return () => {};
    originalResolve = llm.resolveModelInfo.bind(llm);
    llm.resolveModelInfo = async (provider, model, signal) => {
      const info = await originalResolve(provider, model, signal);
      // 对所有文本模型统一声明 image 能力：attachment 预检用
      // `inputModalities` 决定是否放行图片上传，不声明就会被服务端
      // 以 MODEL_DOES_NOT_SUPPORT_IMAGES 拒绝，用户连图都发不出来。
      // 纯文本模型本体（如 DeepSeek）真正收图时会在发送层被
      // llm-pi-ai 的 model.input 校验拒绝，这正好由 Part 3 的
      // agent/request-error 钩子接住：把图片替换为 `[图片N]:"<路径>"`
      // + mmx（MiniMax VLM）识别结果文本后 retry——图片解析任务由
      // mmx 完成，模型本体永远收不到图片字节，只读到识别文本。
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

  // 每个会话的 step 状态：sessionId -> Map(attachmentId -> { path, description? })
  // ⚠️ 必须按会话隔离 + 原地 clear（不能整体重新赋值一个全局 Map）：
  // 多会话并发时，其他 agent 的 pre-step 若共享/重置同一个 Map，会清空
  // 正在 mmx 识别中的图片元数据，导致 request-error 时 imageMeta 为空、
  // 劫持失效（症状：`pi-ai model ... does not support image input` 直接抛给用户）。
  const imageMetaBySession = new Map();

  function metaFor(agent) {
    let sid = "global";
    try {
      const s = agent && agent.session;
      if (s && s.id) sid = String(s.id);
    } catch (_) {
      /* fall through */
    }
    let m = imageMetaBySession.get(sid);
    if (!m) {
      m = new Map();
      imageMetaBySession.set(sid, m);
    }
    return m;
  }

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

  // Part 2a: 把消息里的图片块写盘 + mmx 自动识别，写入会话级 meta。
  // 幂等：同一 attachmentId 已处理过则跳过；识别失败不阻断落盘。
  async function prepareImages(messages, agent, meta, signal) {
    const root = workspaceRootFor(agent);
    const tasks = [];
    for (const message of messages) {
      if (!message || message.role !== "user" || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (!block || block.type !== "image") continue;
        const id = String(
          block.attachment && block.attachment.attachmentId
            ? block.attachment.attachmentId
            : ""
        );
        if (!id || meta.has(id)) continue;
        tasks.push(
          (async () => {
            try {
              const stored = await attachments.readImage(block.attachment, signal);
              const ref = stored.ref;
              const base =
                shaOf(ref) || "img-" + Date.now() + "-" + Math.random().toString(16).slice(2);
              const destPath =
                toPosix(root).replace(/\/+$/, "") + "/.attachments/" + base + extFor(ref.mediaType);
              const ok = await writeImageFile(stored.data, destPath);
              const entry = { path: destPath, description: "" };
              if (ok) {
                meta.set(id, entry);
                // mmx 自动识别（并行，失败不阻断落盘流程）
                if (MMX_SUPPORTED.has(ref.mediaType)) {
                  const result = await runMmxVision(destPath);
                  if (result.ok) entry.description = result.text;
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

  // Part 2: pre-step 落盘 + 识别（保留图片块，客户端渲染缩略图）。
  ctx.on("agent/pre-step", async (payload, next) => {
    const decision = await next();
    console.log("[image-mmx] pre-step: kind=", decision && decision.kind, "hasMessages=", Array.isArray(decision && decision.messages));
    if (!decision || decision.kind !== "enter" || !Array.isArray(decision.messages)) {
      return decision;
    }
    const meSupportsImage = await modelSupportsImages(payload.agent);
    const meImgCount = decision.messages.reduce(function (n, m) {
      if (!m || m.role !== "user" || !Array.isArray(m.content)) return n;
      return n + m.content.filter(function (b) { return b && b.type === "image"; }).length;
    }, 0);
    console.log("[image-mmx] pre-step enter: modelSupportsImages=", meSupportsImage, "图片块数=", meImgCount);
    const meta = metaFor(payload.agent);
    if (meSupportsImage) {
      // 原生模型支持图片：不需要劫持，清掉历史记录
      meta.clear();
    } else {
      // 文本模型：补齐/刷新当前 step 的图片元数据。
      // ⚠️ 不能 clear：历史失败消息里的图片可能仍需要被 Part 3 替换，
      // 否则 pi-ai 会以「不支持图片」反复拒绝整个会话的后续请求。
      await prepareImages(decision.messages, payload.agent, meta, payload.signal);
    }
    return decision;
  });

  function buildTextVersion(message, meta) {
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
        const entry = meta.get(id);
        if (entry && entry.path) {
          let text = '[图片' + idx + ']:"' + entry.path + '"';
          if (entry.description) {
            text += "\n[mmx 视觉识别] " + entry.description;
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

  // 从日志尾部向前扫描会话内「仍在模型可见面（surface）中」的图片消息。
  // 不限于当前 step：历史失败消息里的图片同样需要替换，否则 pi-ai
  // 会以「不支持图片输入」反复拒绝之后的所有请求（会话被卡死）。
  // 用 surface.nodes 过滤已被替换（shadow）的历史消息，避免重复替换报错。
  function findImageMessages(agent) {
    const events = agent && agent.session ? agent.session.events : [];
    let liveSeqs = null;
    try {
      const nodes = agent.session.surface && agent.session.surface.nodes;
      if (Array.isArray(nodes)) liveSeqs = new Set(nodes);
    } catch (_) {
      /* fall through */
    }
    const targets = [];
    for (let i = events.length - 1; i >= 0; i--) {
      const ev = events[i];
      if (ev.type !== "user/message") continue;
      if (liveSeqs && !liveSeqs.has(ev.seq)) continue; // 已被替换，不在模型可见面
      const msg = ev.data;
      if (!msg || !Array.isArray(msg.content)) continue;
      for (const block of msg.content) {
        if (block && block.type === "image") {
          targets.push(ev);
          break;
        }
      }
    }
    targets.reverse();
    return targets;
  }

  // Part 3: 请求被模型输入校验拒绝（如 pi-ai 不支持图片）时，把会话内
  // 所有图片消息替换为路径 + mmx 识别文本后 retry。meta 缺失的图片
  // （如进程重启后残留的历史图片）现场补齐识别——会话可自愈，无需重发。
  ctx.on("agent/request-error", async (payload, next) => {
    const errMsg = payload && payload.failure ? String((payload.failure.message) || payload.failure).slice(0, 120) : "?";
    // 原生支持图片的模型不劫持（避免误替换正常图片消息）
    if (await modelSupportsImages(payload.agent)) return next();
    const meta = metaFor(payload.agent);
    console.log("[image-mmx] request-error: imageMeta.size=", meta.size, "错误=", errMsg);
    const targets = findImageMessages(payload.agent);
    if (targets.length === 0) {
      console.log("[image-mmx] request-error: 无待替换图片消息, 清空 imageMeta 后放行");
      meta.clear();
      return next();
    }
    // 现场补齐缺失的图片元数据（历史残留/重启后丢失）
    const missing = targets.filter((t) =>
      t.data.content.some(
        (b) =>
          b &&
          b.type === "image" &&
          !meta.has(
            String(b.attachment && b.attachment.attachmentId ? b.attachment.attachmentId : "")
          )
      )
    );
    if (missing.length > 0) {
      console.log("[image-mmx] request-error: 现场补齐", missing.length, "条图片的落盘+mmx 识别");
      for (const t of missing) {
        await prepareImages([t.data], payload.agent, meta, payload.signal);
      }
    }
    console.log("[image-mmx] request-error: 将替换", targets.length, "条图片消息后 retry");
    let replaced = 0;
    for (const t of targets) {
      const replacement = buildTextVersion(t.data, meta);
      try {
        payload.agent.session.append("user/message", replacement, {
          surfaceOp: { op: "replace", start: t.seq, end: t.seq },
          sourceEventSeqs: [t.seq],
        });
        replaced += 1;
      } catch (err) {
        console.warn(
          "image-mmx: replace failed:",
          String(err && err.message ? err.message : err)
        );
      }
    }
    meta.clear();
    if (replaced === 0) {
      console.log("[image-mmx] request-error: 无一条替换成功, 放行原错误（防 retry 死循环）");
      return next();
    }
    console.log("[image-mmx] request-error: 替换成功", replaced, "条, retry");
    return { kind: "retry" };
  });
}

export { apply, inject, name };
