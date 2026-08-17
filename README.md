# dsh-image-mmx

> Give DeepSeek Harness **text-only models** a pair of eyes: paste/send images in a conversation, and **mmx (MiniMax VLM)** automatically describes them into the model's context — pure-text models like DeepSeek can understand images without switching models.
>
> 给 DeepSeek Harness 的**文本模型**装一双眼睛：会话里粘贴/发送图片，自动调用 **mmx（MiniMax VLM）** 识别，识别结果注入模型上下文——DeepSeek 等纯文本模型也能理解图片内容，无需切换模型。

---

## English

### What it does

A text-only model (e.g. DeepSeek, `inputModalities` without `image`) normally rejects image attachments with "the current model does not support images". This plugin bridges the gap:

```
You paste an image (text-model session)
   │
   ├─ ① Model capability declaration is patched → server preflight allows the upload
   ├─ ② Image is saved to the session workspace .attachments/ and mmx (MiniMax VLM)
   │     `vision describe` runs automatically
   └─ ③ The image block is replaced, on the model-visible surface only, with
         [图片N]:"<path>" + the mmx recognition text → the text model reads the
         description; the image bytes never reach the model
```

- **Send images from any text model** — no more "model does not support images"
- **Auto recognition**: `mmx vision describe` (MiniMax VLM) extracts elements, layout, and verbatim text (errors / code / UI copy)
- **User side unchanged**: the transcript still renders the original thumbnail (click to enlarge)
- **Graceful degradation**: mmx missing / unauthenticated / timeout (60s) / unsupported format (gif) → falls back to the path text
- **Vision-capable models untouched**: native image input takes the original path, this plugin does nothing

Companion plugin: [dsh-file-bridge](https://github.com/fengs2021/dsh-file-bridge) (📎 attachment button + `send_files` delivery + explorer jump).

### Requirements

`mmx-cli` installed and authenticated (MiniMax official CLI):

```bash
npm install -g mmx-cli
mmx auth login --api-key sk-xxxxx   # or OAuth
mmx vision describe --test.png      # verify it works
```

### Install

Prereqs: DeepSeek Harness `0.1.0-rc.6+`, Node.js 18+.

```bash
dsh plugin --profile web add github:fengs2021/dsh-image-mmx
systemctl restart dsh-web
```

### Usage

1. **Paste / drag an image** into the DSH Web input (or use dsh-file-bridge's 📎 button)
2. The plugin automatically: saves the image → runs mmx recognition → the model reads the description and replies
3. Multiple images: each is recognized independently in parallel, numbered `[图片1] [图片2]...`

### How it works

Built on the [dsh-image-bridge](https://github.com/haitang1/dsh-image-bridge) (MIT) mechanism, adapted:

1. **Capability declaration**: wraps `llm.resolveModelInfo` to declare `image` input for text-only models — the attachment preflight (`MODEL_DOES_NOT_SUPPORT_IMAGES`) passes, so the image can be uploaded at all
2. **agent/pre-step**: image is saved to disk + `mmx vision describe` runs in parallel
3. **agent/request-error**: the image block is replaced on the model-visible surface with path + recognition text, then retried — **the model never receives image bytes; mmx owns the recognition task**

### Notes

- The model-visible replacement is a *surface* change: your transcript keeps the original thumbnail
- DeepSeek models keep their real (text-only) capability declaration — the plugin does not claim fake image support; it *routes* the recognition to mmx instead

## 中文

### 功能

- **图片收发**：文本模型（`inputModalities` 无 image）也能发送含图消息，不再报「当前模型不支持图片」
- **自动识别**：图片落盘到会话工作区 `.attachments/`，自动执行 `mmx vision describe`（MiniMax VLM）
- **模型可见替换**：图片块在模型可见面替换为 `[图片N]:"<路径>"` + mmx 识别文本（画面元素、布局、报错/代码/界面文字逐字保留）
- **用户侧不变**：人类 transcript 照常渲染原图缩略图（可点击放大）
- **优雅降级**：mmx 未安装/未认证/超时（60s）/格式不支持（gif）时降级为路径文本
- **原生模型免打扰**：本身支持图片输入的模型走原生路径，不做任何处理

### 安装

前置：DeepSeek Harness `0.1.0-rc.6+`、Node.js 18+、已安装并认证 `mmx-cli`。

```bash
dsh plugin --profile web add github:fengs2021/dsh-image-mmx
systemctl restart dsh-web
```

### 使用

1. 在 DSH Web 输入框**粘贴/拖入图片**（或配合 dsh-file-bridge 的 📎 附件按钮选图）
2. 发送后插件自动：落盘 → mmx 识别 → 模型直接读到识别结果并回复
3. 多图支持：每张图独立并行识别，按 `[图片1] [图片2]...` 编号

### 实现

基于 [dsh-image-bridge](https://github.com/haitang1/dsh-image-bridge)（MIT）机制改造：

1. 包装 `llm.resolveModelInfo` 为文本模型补 `image` 模态（放行附件预检，否则图片连发都发不出去）
2. `agent/pre-step`：图片落盘 + 并行调用 `mmx vision describe`
3. `agent/request-error`：surface replace 把图片块换成识别文本，`retry` 重发——**模型本体永远收不到图片字节，解析任务由 mmx 完成**

### 说明

- 替换只发生在「模型可见面」：你的对话记录里始终是原图缩略图
- DeepSeek 等模型的能力声明保持真实（不注入虚假图片能力），识别由 mmx 路由完成

## License

MIT