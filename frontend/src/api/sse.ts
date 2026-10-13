import type { PipelineEvent, RunRequest } from "./types";

/**
 * 消费后端 SSE（POST /api/qa/run）。
 * EventSource 不支持 POST，因此用 fetch + ReadableStream 手工解析。
 */
export async function streamSSE(
  url: string,
  body: RunRequest,
  onEvent: (ev: PipelineEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!resp.ok || !resp.body) {
    let detail = `HTTP ${resp.status}`;
    try {
      detail += " " + (await resp.text()).slice(0, 200);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      // 正常关闭：释放 reader，避免浏览器把连接关闭记成 abort 红字
      try {
        await reader.cancel();
      } catch {
        /* ignore */
      }
      break;
    }
    buf += decoder.decode(value, { stream: true });
    // SSE 事件以空行分隔
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const dataLine = raw
        .split("\n")
        .find((l) => l.startsWith("data: "));
      if (!dataLine) continue;
      try {
        onEvent(JSON.parse(dataLine.slice(6)) as PipelineEvent);
      } catch {
        // 忽略无法解析的行，保证流不中断
      }
    }
  }
}
