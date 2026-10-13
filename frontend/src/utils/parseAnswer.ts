import type { AnswerSection } from "../api/types";

/** 前端版【Tag】分段解析（与后端 dataset.py _parse_sections 逻辑一致）。 */
export function parseAnswer(answer: string): AnswerSection[] {
  const re = /【([^】]+)】/g;
  const marks: { title: string; headStart: number; bodyStart: number }[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(answer)) !== null) {
    marks.push({ title: m[1], headStart: m.index, bodyStart: m.index + m[0].length });
  }
  if (marks.length === 0) return [];
  return marks
    .map((mark, i) => {
      const bodyEnd = i + 1 < marks.length ? marks[i + 1].headStart : answer.length;
      return { title: mark.title, text: answer.slice(mark.bodyStart, bodyEnd).trim() };
    })
    .filter((x) => x.text.length > 0);
}
